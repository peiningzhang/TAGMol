import argparse
import os
import multiprocessing as mp

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter, defaultdict

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils.evaluation.similarity import mean_pairwise_tanimoto
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def _docking_worker(payload, queue):
    """Run docking in child process to isolate native-library crashes."""
    try:
        mol = Chem.MolFromMolBlock(payload['mol_block'], removeHs=False)
        if mol is None:
            raise ValueError('Failed to deserialize ligand mol block')
        mode = payload['docking_mode']
        ligand_filename = payload['ligand_filename']
        protein_root = payload['protein_root']
        protein_filename = payload['protein_filename']
        exhaustiveness = payload['exhaustiveness']

        if mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, ligand_filename=ligand_filename, protein_root=protein_root, protein_filename=protein_filename
            )
            vina_results = vina_task.run_sync()
        elif mode in ['vina_score', 'vina_dock']:
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename=ligand_filename, protein_root=protein_root, protein_filename=protein_filename
            )
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
            vina_results = {'score_only': score_only_results, 'minimize': minimize_results}
            if mode == 'vina_dock':
                docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                vina_results['dock'] = docking_results
        else:
            vina_results = None
        queue.put(('ok', vina_results))
    except Exception as e:
        queue.put(('err', repr(e)))


def _run_docking_isolated(mol, docking_mode, ligand_filename, protein_root, protein_filename, exhaustiveness):
    # Linux default: fork is lightweight and avoids repeated interpreter cold start.
    ctx = mp.get_context('fork')
    queue = ctx.Queue(maxsize=1)
    payload = {
        'mol_block': Chem.MolToMolBlock(mol),
        'docking_mode': docking_mode,
        'ligand_filename': ligand_filename,
        'protein_root': protein_root,
        'protein_filename': protein_filename,
        'exhaustiveness': exhaustiveness,
    }
    proc = ctx.Process(target=_docking_worker, args=(payload, queue))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f'docking subprocess crashed (exit_code={proc.exitcode})')
    if queue.empty():
        raise RuntimeError('docking subprocess returned no result')
    status, value = queue.get()
    if status == 'ok':
        return value
    raise RuntimeError(f'docking subprocess error: {value}')


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    if not all_ring_sizes:
        logger.info('ring size: (no evaluated mols, skip ratio)')
        return
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def run_evaluation(sample_path, eval_step=-1, eval_num_examples=None, docking_mode='none',
                   protein_root='./data/crossdocked_v1.1_rmsd1.0', atom_enc_mode='add_aromatic',
                   verbose=False, save=True, exhaustiveness=16, docking_isolate_process=False, logger=None):
    """
    Run full evaluation on generated samples in sample_path. Returns a flat dict of metrics
    suitable for logging (e.g. wandb). Can be called from train_diffusion for periodic eval.
    """
    result_path = os.path.join(sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    if logger is None:
        logger = misc.get_logger('evaluate', log_dir=result_path)
    if not verbose:
        RDLogger.DisableLog('rdApp.*')

    results_fn_list = glob(os.path.join(sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if eval_num_examples is not None:
        results_fn_list = results_fn_list[:eval_num_examples]
    num_examples = len(results_fn_list)

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    first_docking_error = [None]
    _debug_first_sample_logged = [False]
    all_pair_dist, all_bond_dist = [], []
    success_pair_dist, success_atom_types = [], Counter()
    for _, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[eval_step], pred_v[eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                continue
            n_recon_success += 1
            if '.' in smiles:
                continue
            n_complete += 1
            try:
                chem_results = scoring_func.get_chem(mol)
                data = r.get('data')
                ligand_filename = r.get('ligand_filename')
                protein_filename = r.get('protein_filename')
                if ligand_filename is None or protein_filename is None:
                    if data is not None:
                        lf = data.get('ligand_filename', None) if isinstance(data, dict) else getattr(data, 'ligand_filename', None)
                        pf = data.get('protein_filename', None) if isinstance(data, dict) else getattr(data, 'protein_filename', None)
                        if ligand_filename is None:
                            ligand_filename = lf
                        if protein_filename is None:
                            protein_filename = pf
                if not _debug_first_sample_logged[0] and docking_mode != 'none':
                    _debug_first_sample_logged[0] = True
                    protein_path = os.path.join(protein_root, protein_filename) if protein_filename else None
                    logger.info(f"[Docking debug] ligand_filename={ligand_filename!r}, protein_filename={protein_filename!r}, protein_path={protein_path!r}, exists={os.path.exists(protein_path) if protein_path else 'N/A'}")
                vina_results = None
                if docking_mode == 'qvina':
                    if ligand_filename is None and protein_filename is None:
                        data_keys = list(data.keys()) if isinstance(data, dict) else [k for k in dir(data) if not k.startswith('_')]
                        raise ValueError(f"data must have ligand_filename or protein_filename for docking. data keys: {data_keys[:20]}")
                    if docking_isolate_process:
                        vina_results = _run_docking_isolated(
                            mol=mol, docking_mode=docking_mode, ligand_filename=ligand_filename,
                            protein_root=protein_root, protein_filename=protein_filename, exhaustiveness=exhaustiveness
                        )
                    else:
                        vina_task = QVinaDockingTask.from_generated_mol(
                            mol, ligand_filename=ligand_filename, protein_root=protein_root,
                            protein_filename=protein_filename)
                        vina_results = vina_task.run_sync()
                elif docking_mode in ['vina_score', 'vina_dock']:
                    if ligand_filename is None and protein_filename is None:
                        data_keys = list(data.keys()) if isinstance(data, dict) else [k for k in dir(data) if not k.startswith('_')]
                        raise ValueError(f"data must have ligand_filename or protein_filename for docking. data keys: {data_keys[:20]}")
                    if docking_isolate_process:
                        vina_results = _run_docking_isolated(
                            mol=mol, docking_mode=docking_mode, ligand_filename=ligand_filename,
                            protein_root=protein_root, protein_filename=protein_filename, exhaustiveness=exhaustiveness
                        )
                    else:
                        vina_task = VinaDockingTask.from_generated_mol(
                            mol, ligand_filename=ligand_filename, protein_root=protein_root,
                            protein_filename=protein_filename)
                        score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
                        minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
                        vina_results = {'score_only': score_only_results, 'minimize': minimize_results}
                        if docking_mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                            vina_results['dock'] = docking_results
                else:
                    vina_results = None
                n_eval_success += 1
            except Exception as e:
                if first_docking_error[0] is None:
                    first_docking_error[0] = (r_name, e)
                if verbose:
                    logger.info(f"Docking failed for sample: {e}")
                continue
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist
            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)
            results.append({
                'mol': mol,
                'smiles': smiles,
                'chem_results': chem_results,
                'vina': vina_results,
                'ligand_filename': ligand_filename,
                'protein_filename': protein_filename,
            })

    if docking_mode != 'none' and n_eval_success == 0 and first_docking_error[0] is not None:
        r_name, err = first_docking_error[0]
        msg = f"All docking failed (eval_success=0). First error from {os.path.basename(r_name)}: {err}"
        logger.warning(msg)
        print(msg, flush=True)

    fraction_mol_stable = all_mol_stable / num_samples if num_samples else 0.0
    fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom > 0 else 0.0
    fraction_recon = n_recon_success / num_samples if num_samples else 0.0
    fraction_eval = n_eval_success / num_samples if num_samples else 0.0
    fraction_complete = n_complete / num_samples if num_samples else 0.0

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    if len(success_pair_dist) > 0:
        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    else:
        success_js_metrics = {}
        success_pair_length_profile = None
    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types) if sum(success_atom_types.values()) > 0 else None

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    qed_mean = float(np.mean(qed)) if qed else None
    qed_med = float(np.median(qed)) if qed else None
    sa_mean = float(np.mean(sa)) if sa else None
    sa_med = float(np.median(sa)) if sa else None

    out = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete,
        'n_recon': n_recon_success,
        'n_complete': n_complete,
        'n_eval': len(results),
        'n_samples': num_samples,
        'atom_type_js': atom_type_js,
        'QED_mean': qed_mean,
        'QED_med': qed_med,
        'SA_mean': sa_mean,
        'SA_med': sa_med,
    }
    for k, v in c_bond_length_dict.items():
        out[k] = v
    for k, v in success_js_metrics.items():
        out[k] = v

    # Vina metrics (for vina_score / vina_dock) so callers (e.g. train quick_eval) get them in out
    try:
        if results and docking_mode in ['vina_score', 'vina_dock']:
            vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
            vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
            out['Vina_score_mean'] = float(np.mean(vina_score_only))
            out['Vina_score_med'] = float(np.median(vina_score_only))
            out['Vina_min_mean'] = float(np.mean(vina_min))
            out['Vina_min_med'] = float(np.median(vina_min))
        else:
            out['Vina_score_mean'] = out['Vina_score_med'] = out['Vina_min_mean'] = out['Vina_min_med'] = None
    except Exception as e:
        logger.warning(f"Error calculating Vina metrics: {e}")
        out['Vina_score_mean'] = out['Vina_score_med'] = out['Vina_min_mean'] = out['Vina_min_med'] = None

    # Per-pocket: mean pairwise RDK Tanimoto among generated mols; Diversity = 1 - that mean (macro avg over pockets).
    try:
        pocket_key_to_mols = defaultdict(list)
        for r in results:
            pk = r.get('protein_filename') or r.get('ligand_filename') or 'unknown'
            pocket_key_to_mols[pk].append(r['mol'])
        pocket_mean_sims = []
        pocket_divs = []
        for mols in pocket_key_to_mols.values():
            mp = mean_pairwise_tanimoto(mols)
            if mp is not None:
                pocket_mean_sims.append(mp)
                pocket_divs.append(1.0 - mp)
        if pocket_mean_sims:
            out['Mean_pairwise_Tanimoto'] = float(np.mean(pocket_mean_sims))
            out['Diversity'] = float(np.mean(pocket_divs))
            out['Diversity_med'] = float(np.median(pocket_divs))
        else:
            out['Mean_pairwise_Tanimoto'] = None
            out['Diversity'] = None
            out['Diversity_med'] = None
    except Exception as e:
        logger.warning(f"Error calculating RDK fingerprint diversity: {e}")
        out['Mean_pairwise_Tanimoto'] = None
        out['Diversity'] = None
        out['Diversity_med'] = None

    if save:
        validity_dict = {k: out[k] for k in ['mol_stable', 'atm_stable', 'recon_success', 'eval_success', 'complete']}
        torch.save({'stability': validity_dict, 'bond_length': all_bond_dist, 'all_results': results},
                   os.path.join(result_path, f'metrics_{eval_step}.pt'))
        if success_pair_length_profile is not None:
            eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                                metrics=success_js_metrics,
                                                save_path=os.path.join(result_path, f'pair_dist_hist_{eval_step}.png'))
    return out, results


def metrics_one_line_tsv_parts(out, results, docking_mode):
    """Tab-separated field strings matching CLI --one_line (uses '\\t ' between columns)."""
    bond_js_keys = [k for k in out if k.startswith('JSD_') and k not in ('JSD_CC_2A', 'JSD_All_12A')]
    pair_js_keys = [k for k in ('JSD_CC_2A', 'JSD_All_12A') if k in out]
    c_bond_length_dict = {k: out[k] for k in bond_js_keys}
    success_js_metrics = {k: out[k] for k in pair_js_keys}

    def _fmt(v):
        if v is None:
            return 'N/A'
        if isinstance(v, float):
            return '%.4f' % v
        return str(v)

    names, values = [], []
    for name in ['mol_stable', 'atm_stable', 'recon_success', 'eval_success', 'complete']:
        names.append(name)
        values.append(_fmt(out[name]))
    for k in sorted(c_bond_length_dict.keys()):
        names.append(k)
        values.append(_fmt(out.get(k)))
    for k in sorted(success_js_metrics.keys()):
        names.append(k)
        values.append(_fmt(out.get(k)))
    names.append('atom_type_js')
    values.append(_fmt(out.get('atom_type_js')))
    names.extend(['n_recon', 'n_complete', 'n_eval'])
    values.extend([str(out['n_recon']), str(out['n_complete']), str(out['n_eval'])])
    names.extend(['QED_mean', 'QED_med', 'SA_mean', 'SA_med'])
    values.extend([_fmt(out['QED_mean']), _fmt(out['QED_med']), _fmt(out['SA_mean']), _fmt(out['SA_med'])])
    names.extend(['Mean_pairwise_Tanimoto', 'Diversity', 'Diversity_med'])
    values.extend(
        [
            _fmt(out.get('Mean_pairwise_Tanimoto')),
            _fmt(out.get('Diversity')),
            _fmt(out.get('Diversity_med')),
        ]
    )
    if docking_mode == 'qvina' and results:
        vina = [r['vina'][0]['affinity'] for r in results]
        names.extend(['Vina_mean', 'Vina_med'])
        values.extend([_fmt(np.mean(vina)), _fmt(np.median(vina))])
    elif docking_mode in ['vina_dock', 'vina_score'] and results:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        names.extend(['Vina_score_mean', 'Vina_score_med', 'Vina_min_mean', 'Vina_min_med'])
        values.extend(
            [
                _fmt(np.mean(vina_score_only)),
                _fmt(np.median(vina_score_only)),
                _fmt(np.mean(vina_min)),
                _fmt(np.median(vina_min)),
            ]
        )
    sep = '\t '
    return sep.join(names), sep.join(values)


def report_evaluation_to_logger(out, results, docking_mode, logger, one_line=False):
    """Log the same blocks as CLI evaluate_diffusion after run_evaluation. If one_line, also log METRICS_ONE_LINE_*."""
    validity_dict = {k: out[k] for k in ['mol_stable', 'atm_stable', 'recon_success', 'eval_success', 'complete']}
    print_dict(validity_dict, logger)

    bond_js_keys = [k for k in out if k.startswith('JSD_') and k not in ('JSD_CC_2A', 'JSD_All_12A')]
    pair_js_keys = [k for k in ('JSD_CC_2A', 'JSD_All_12A') if k in out]
    c_bond_length_dict = {k: out[k] for k in bond_js_keys}
    success_js_metrics = {k: out[k] for k in pair_js_keys}
    if c_bond_length_dict:
        logger.info('JS bond distances of complete mols: ')
        print_dict(c_bond_length_dict, logger)
    if success_js_metrics:
        print_dict(success_js_metrics, logger)

    logger.info('Atom type JS: %s' % out.get('atom_type_js'))
    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        out['n_recon'], out['n_complete'], out['n_eval']))

    qed = [r['chem_results']['qed'] for r in results]
    if qed:
        logger.info('QED:   Mean: %.3f Median: %.3f' % (out['QED_mean'], out['QED_med']))
        logger.info('SA:    Mean: %.3f Median: %.3f' % (out['SA_mean'], out['SA_med']))
    else:
        logger.info('QED:   Mean: None Median: None')
        logger.info('SA:    Mean: None Median: None')
    if out.get('Diversity') is not None:
        logger.info(
            'Diversity (RDK fp): Mean pairwise Tanimoto %.4f | Diversity mean: %.4f | Diversity median: %.4f'
            % (out['Mean_pairwise_Tanimoto'], out['Diversity'], out['Diversity_med'])
        )
    else:
        logger.info('Diversity (RDK fp): N/A (need >=2 mols per pocket with valid fps)')
    if docking_mode == 'qvina' and results:
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif docking_mode in ['vina_dock', 'vina_score'] and results:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if one_line:
        names_tsv, vals_tsv = metrics_one_line_tsv_parts(out, results, docking_mode)
        logger.info('METRICS_ONE_LINE_HEAD\t' + names_tsv)
        logger.info('METRICS_ONE_LINE_VAL\t' + vals_tsv)
        return names_tsv, vals_tsv
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='none', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--docking_isolate_process', type=eval, default=False,
                        help='Run each docking in a child process to avoid whole-run abort from native library crashes.')
    parser.add_argument('--one_line', action='store_true', help='Print all metrics in one line at the end (no ring size)')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)

    out, results = run_evaluation(
        sample_path=args.sample_path,
        eval_step=args.eval_step,
        eval_num_examples=args.eval_num_examples,
        docking_mode=args.docking_mode,
        protein_root=args.protein_root,
        atom_enc_mode=args.atom_enc_mode,
        verbose=args.verbose,
        save=args.save,
        exhaustiveness=args.exhaustiveness,
        docking_isolate_process=args.docking_isolate_process,
        logger=logger,
    )
    report_evaluation_to_logger(out, results, args.docking_mode, logger, one_line=args.one_line)
