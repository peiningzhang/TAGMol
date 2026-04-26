"""
Merge eval_shards/*/eval_results/metrics_*.pt from a quick_eval / sample directory into one
eval_results/metrics_<step>.pt with the same schema as evaluate_diffusion, and global stats
matching a single full run.

Requires: parent sample_path with result_*.pt and eval_shards/shard_*/eval_results/metrics_*.pt

Usage:
  cd /path/to/TAGMol && python scripts/merge_eval_diffusion_shards.py \\
    --sample_path tmp_eval/quick_eval_...
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob

import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm

# repo root on path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.evaluation import eval_atom_type, analyze, eval_bond_length
from utils.evaluation.similarity import mean_pairwise_tanimoto
from utils import misc, reconstruct, transforms
from scripts.evaluate_diffusion import metrics_one_line_tsv_parts, report_evaluation_to_logger


def _scan_denominators(sample_path, eval_step, atom_enc_mode):
    """Same stability / recon / complete / num_samples accounting as evaluate_diffusion (no docking)."""
    results_fn_list = glob(os.path.join(sample_path, "*result_*.pt"))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split("_")[-1]))
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_complete = 0, 0
    for r_name in tqdm(results_fn_list, desc="Scan result_*.pt"):
        r = torch.load(r_name)
        all_pred_ligand_pos = r["pred_ligand_pos_traj"]
        all_pred_ligand_v = r["pred_ligand_v_traj"]
        num_samples += len(all_pred_ligand_pos)
        for pred_pos, pred_v in zip(all_pred_ligand_pos, all_pred_ligand_v):
            pred_pos, pred_v = pred_pos[eval_step], pred_v[eval_step]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                continue
            n_recon_success += 1
            if "." in smiles:
                continue
            n_complete += 1
    return {
        "num_samples": num_samples,
        "all_mol_stable": all_mol_stable,
        "all_atom_stable": all_atom_stable,
        "all_n_atom": all_n_atom,
        "n_recon": n_recon_success,
        "n_complete": n_complete,
    }


def _pair_dist_from_mol(mol):
    pos = mol.GetConformer().GetPositions()
    elements = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())], dtype=np.int64)
    return eval_bond_length.pair_distance_from_pos_v(pos, elements)


def _merged_out_from_shards(sample_path, eval_step, docking_mode, atom_enc_mode, den):
    shard_glob = os.path.join(sample_path, "eval_shards", "shard_*", "eval_results", "metrics_%s.pt" % eval_step)
    paths = glob(shard_glob)
    if not paths:
        raise FileNotFoundError("No shard metrics found: %s" % shard_glob)

    def shard_key(p):
        m = re.search(r"shard_(\d+)_(\d+)", p)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    paths = sorted(paths, key=shard_key)
    merged_results = []
    merged_bond = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        ar = d.get("all_results", [])
        merged_results.extend(ar)
        merged_bond.extend(d.get("bond_length", []))

    num_samples = den["num_samples"]
    n_recon = den["n_recon"]
    n_complete = den["n_complete"]
    n_eval = len(merged_results)

    fraction_mol_stable = den["all_mol_stable"] / num_samples if num_samples else 0.0
    fraction_atm_stable = den["all_atom_stable"] / den["all_n_atom"] if den["all_n_atom"] > 0 else 0.0
    fraction_recon = n_recon / num_samples if num_samples else 0.0
    fraction_eval = n_eval / num_samples if num_samples else 0.0
    fraction_complete = n_complete / num_samples if num_samples else 0.0

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(merged_bond)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)

    success_pair_dist = []
    success_atom_types = Counter()
    for r in merged_results:
        mol = r["mol"]
        success_pair_dist += _pair_dist_from_mol(mol)
        for z in [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]:
            success_atom_types[z] += 1

    if len(success_pair_dist) > 0:
        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    else:
        success_js_metrics = {}
        success_pair_length_profile = None

    atom_type_js = (
        eval_atom_type.eval_atom_type_distribution(success_atom_types) if sum(success_atom_types.values()) > 0 else None
    )

    qed = [r["chem_results"]["qed"] for r in merged_results]
    sa = [r["chem_results"]["sa"] for r in merged_results]
    qed_mean = float(np.mean(qed)) if qed else None
    qed_med = float(np.median(qed)) if qed else None
    sa_mean = float(np.mean(sa)) if sa else None
    sa_med = float(np.median(sa)) if sa else None

    out = {
        "mol_stable": fraction_mol_stable,
        "atm_stable": fraction_atm_stable,
        "recon_success": fraction_recon,
        "eval_success": fraction_eval,
        "complete": fraction_complete,
        "n_recon": n_recon,
        "n_complete": n_complete,
        "n_eval": n_eval,
        "n_samples": num_samples,
        "atom_type_js": atom_type_js,
        "QED_mean": qed_mean,
        "QED_med": qed_med,
        "SA_mean": sa_mean,
        "SA_med": sa_med,
    }
    for k, v in c_bond_length_dict.items():
        out[k] = v
    for k, v in success_js_metrics.items():
        out[k] = v

    try:
        if merged_results and docking_mode in ("vina_score", "vina_dock"):
            vina_score_only = [r["vina"]["score_only"][0]["affinity"] for r in merged_results]
            vina_min = [r["vina"]["minimize"][0]["affinity"] for r in merged_results]
            out["Vina_score_mean"] = float(np.mean(vina_score_only))
            out["Vina_score_med"] = float(np.median(vina_score_only))
            out["Vina_min_mean"] = float(np.mean(vina_min))
            out["Vina_min_med"] = float(np.median(vina_min))
        else:
            out["Vina_score_mean"] = out["Vina_score_med"] = out["Vina_min_mean"] = out["Vina_min_med"] = None
        if merged_results and docking_mode == "vina_dock":
            vina_dock_aff = [r["vina"]["dock"][0]["affinity"] for r in merged_results]
            out["Vina_dock_mean"] = float(np.mean(vina_dock_aff))
            out["Vina_dock_med"] = float(np.median(vina_dock_aff))
        else:
            out["Vina_dock_mean"] = out["Vina_dock_med"] = None
    except Exception as e:  # noqa: BLE001
        print("Warning: Vina aggregate failed: %s" % e)
        out["Vina_score_mean"] = out["Vina_score_med"] = out["Vina_min_mean"] = out["Vina_min_med"] = None
        out["Vina_dock_mean"] = out["Vina_dock_med"] = None

    try:
        pocket_key_to_mols = defaultdict(list)
        for r in merged_results:
            pk = r.get("protein_filename") or r.get("ligand_filename") or "unknown"
            pocket_key_to_mols[pk].append(r["mol"])
        pocket_mean_sims = []
        pocket_divs = []
        for mols in pocket_key_to_mols.values():
            mp = mean_pairwise_tanimoto(mols)
            if mp is not None:
                pocket_mean_sims.append(mp)
                pocket_divs.append(1.0 - mp)
        if pocket_mean_sims:
            out["Mean_pairwise_Tanimoto"] = float(np.mean(pocket_mean_sims))
            out["Diversity"] = float(np.mean(pocket_divs))
            out["Diversity_med"] = float(np.median(pocket_divs))
        else:
            out["Mean_pairwise_Tanimoto"] = out["Diversity"] = out["Diversity_med"] = None
    except Exception as e:  # noqa: BLE001
        print("Warning: diversity metrics failed: %s" % e)
        out["Mean_pairwise_Tanimoto"] = out["Diversity"] = out["Diversity_med"] = None

    return out, merged_results, merged_bond, success_pair_length_profile, success_js_metrics, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True, help="Parent dir with result_*.pt and eval_shards/")
    parser.add_argument("--eval_step", type=int, default=-1)
    parser.add_argument(
        "--docking_mode", type=str, default="vina_dock", choices=["qvina", "vina_score", "vina_dock", "none"]
    )
    parser.add_argument("--atom_enc_mode", type=str, default="add_aromatic")
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip pair_dist_hist_*.png (default: write if JSD pair metrics exist).",
    )
    args = parser.parse_args()
    sample_path = os.path.abspath(args.sample_path)
    if not os.path.isdir(sample_path):
        raise SystemExit("Not a directory: %s" % sample_path)

    outdir = os.path.join(sample_path, "eval_results")
    os.makedirs(outdir, exist_ok=True)
    logger = misc.get_logger("merge_eval", log_dir=outdir)

    print("Scanning all result_*.pt for stability / n_samples ...")
    den = _scan_denominators(sample_path, args.eval_step, args.atom_enc_mode)
    print("num_samples=%d  n_recon=%d  n_complete=%d" % (den["num_samples"], den["n_recon"], den["n_complete"]))

    out, merged_results, merged_bond, success_pair_len_prof, success_js, shard_paths = _merged_out_from_shards(
        sample_path, args.eval_step, args.docking_mode, args.atom_enc_mode, den
    )
    print("Merged %d shard file(s), %d evaluated molecules in all_results" % (len(shard_paths), len(merged_results)))

    validity_dict = {k: out[k] for k in ["mol_stable", "atm_stable", "recon_success", "eval_success", "complete"]}
    out_path = os.path.join(outdir, "metrics_%d.pt" % args.eval_step)
    torch.save(
        {
            "stability": validity_dict,
            "bond_length": merged_bond,
            "all_results": merged_results,
            "merge_meta": {
                "shard_files": [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in shard_paths],
                "n_shards": len(shard_paths),
            },
        },
        out_path,
    )
    print("Wrote: %s" % out_path)

    if not args.no_plot and success_pair_len_prof is not None and success_js:
        hist_path = os.path.join(outdir, "pair_dist_hist_%d.png" % args.eval_step)
        eval_bond_length.plot_distance_hist(
            success_pair_len_prof, metrics=success_js, save_path=hist_path
        )
        print("Wrote: %s" % hist_path)

    report_evaluation_to_logger(out, merged_results, args.docking_mode, logger, one_line=True)
    n_tsv, v_tsv = metrics_one_line_tsv_parts(out, merged_results, args.docking_mode)
    print("METRICS_ONE_LINE_HEAD\t" + n_tsv)
    print("METRICS_ONE_LINE_VAL\t" + v_tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
