import argparse
import glob
import json
import os
import subprocess
import sys
import time
import tempfile
from collections import defaultdict
from datetime import datetime

import torch
import yaml
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from models.molopt_guide_model import DockGuideNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
# Multi‑guide diffusion function
from scripts.sample_multi_guided_diffusion import sample_guided_diffusion_ligand as sample_multi_guided_diffusion
from scripts.evaluate_diffusion import report_evaluation_to_logger, run_evaluation


def export_metrics_to_grouped_sdf(pt_file, out_dir, logger):
    """Write one multi-molecule SDF per pocket from evaluate_diffusion metrics_*.pt."""
    try:
        from rdkit import Chem
    except ImportError as e:
        logger.error("RDKit is required for --export_grouped_sdf: %s", e)
        return False
    if not os.path.isfile(pt_file):
        logger.error("Metrics file not found (run evaluation with --save first): %s", pt_file)
        return False
    os.makedirs(out_dir, exist_ok=True)
    data = torch.load(pt_file)
    results = data.get("all_results", [])
    grouped_mols = defaultdict(list)
    for i, res in enumerate(results):
        mol = res.get("mol")
        if mol is None:
            continue
        target_name = res.get("ligand_filename", "target_%d" % i)
        if isinstance(target_name, str):
            if target_name.startswith("LIGAND_"):
                target_name = target_name.replace("LIGAND_", "")
            target_name_base = target_name.split(".")[0].replace("/", "_")
        else:
            target_name_base = "target_%d" % i
        if "chem_results" in res and isinstance(res["chem_results"], dict):
            cr = res["chem_results"]
            if "qed" in cr:
                mol.SetProp("QED", str(cr["qed"]))
            if "sa" in cr:
                mol.SetProp("SA", str(cr["sa"]))
        if res.get("vina"):
            try:
                val = res["vina"].get("score_only", [{}])[0].get("affinity", "")
                mol.SetProp("Vina", str(val))
            except (TypeError, IndexError, KeyError):
                pass
        grouped_mols[target_name_base].append(mol)
    n_files, n_mols = 0, 0
    for target_name, mols in grouped_mols.items():
        sdf_path = os.path.join(out_dir, "%s_generated.sdf" % target_name)
        w = Chem.SDWriter(sdf_path)
        for m in mols:
            w.write(m)
            n_mols += 1
        w.close()
        n_files += 1
    logger.info(
        "Exported %d molecules into %d grouped SDF files under %s",
        n_mols,
        n_files,
        out_dir,
    )
    return n_files > 0


def run_genbench3d_suite(
    sdf_group_dir,
    gb3d_dir,
    test_set_dir,
    genbench_python,
    logger,
    do_conf_analysis=False,
    no_vina=False,
):
    """Run GenBench3D sb_benchmark_mols.py on each *_generated.sdf and write aggregated JSON.

    With do_conf_analysis=True, passes --do_conf_analysis so GenBench3D also computes Validity3D
    (KDE vs reference geometry), TFD-based Uniqueness3D/Diversity3D/Novelty3D, and MMFF strain energy.
    Use -s ligboundconf (default) and ligboundconf_path in GenBench default.yaml — no CSD license needed.

    With no_vina=True, passes --no_vina (requires a patched sb_benchmark_mols.py that implements this flag).
    Steric clash and distance-to-native-centroid still use the pocket; Vina / relative Vina metrics are omitted.
    """
    # subprocess uses cwd=gb3d_dir; all paths passed to sb_benchmark_mols.py must be absolute.
    sdf_group_dir = os.path.abspath(sdf_group_dir)
    gb3d_dir = os.path.abspath(gb3d_dir)
    test_set_dir = os.path.abspath(test_set_dir)
    output_dir = os.path.join(sdf_group_dir, "genbench_results")
    os.makedirs(output_dir, exist_ok=True)
    sdf_files = glob.glob(os.path.join(sdf_group_dir, "*_generated.sdf"))
    if not sdf_files:
        logger.error("No *_generated.sdf under %s; use --export_grouped_sdf or point --sdf_grouped_dir to existing SDFs.", sdf_group_dir)
        return False
    all_native = glob.glob(os.path.join(test_set_dir, "*", "*.sdf"))
    ligand_map = {}
    for np_path in all_native:
        lig_base = os.path.basename(np_path).replace(".sdf", "")
        pocket_dir_name = os.path.basename(os.path.dirname(np_path))
        ligand_map["%s_%s" % (pocket_dir_name, lig_base)] = np_path
    failed = []
    success = 0
    sb_script = os.path.join(gb3d_dir, "sb_benchmark_mols.py")
    cfg_yaml = os.path.join(gb3d_dir, "config", "default.yaml")
    if not os.path.isfile(sb_script) or not os.path.isfile(cfg_yaml):
        logger.error("GenBench3D not found at gb3d_dir=%s (missing sb_benchmark_mols.py or config/default.yaml)", gb3d_dir)
        return False
    for sdf in sdf_files:
        basename = os.path.basename(sdf).replace("_generated.sdf", "")
        native_ligand = ligand_map.get(basename)
        if not native_ligand:
            logger.warning("Native ligand not found in test_set for %s", basename)
            failed.append(basename)
            continue
        pocket_dir = os.path.dirname(native_ligand)
        proteins = [f for f in os.listdir(pocket_dir) if f.endswith("_rec.pdb")]
        if not proteins:
            logger.warning("Protein not found in %s", pocket_dir)
            failed.append(basename)
            continue
        protein_pdb = os.path.abspath(os.path.join(pocket_dir, proteins[0]))
        sdf_abs = os.path.abspath(sdf)
        native_abs = os.path.abspath(native_ligand)
        out_json = os.path.abspath(os.path.join(output_dir, "results_%s.json" % basename))
        cmd = [
            genbench_python,
            sb_script,
            "-c",
            cfg_yaml,
            "-i",
            sdf_abs,
            "-p",
            protein_pdb,
            "-n",
            native_abs,
            "-o",
            out_json,
            "-s",
            "ligboundconf",
        ]
        if do_conf_analysis:
            cmd.append("--do_conf_analysis")
        if no_vina:
            cmd.append("--no_vina")
        logger.info("GenBench3D: %s", basename)
        try:
            subprocess.run(cmd, cwd=gb3d_dir, check=True)
            success += 1
        except subprocess.CalledProcessError as e:
            logger.warning("GenBench3D failed for %s: %s", basename, e)
            failed.append(basename)
    logger.info("GenBench3D finished: success=%d failed=%d; results in %s", success, len(failed), output_dir)
    merged = []
    for jf in sorted(glob.glob(os.path.join(output_dir, "results_*.json"))):
        try:
            with open(jf, "r") as f:
                res = json.load(f)
            res["pocket"] = os.path.basename(jf)
            merged.append(res)
        except (json.JSONDecodeError, OSError):
            pass
    if merged:
        agg = os.path.join(output_dir, "all_results_aggregated.json")
        with open(agg, "w") as outf:
            json.dump(merged, outf)
        logger.info("Wrote aggregated GenBench3D JSON: %s", agg)
    return success > 0 or len(merged) > 0


def print_genbench_console_report(aggregated_json_path, logger):
    """Same table as: python get_genbench_report.py <all_results_aggregated.json>"""
    aggregated_json_path = os.path.abspath(aggregated_json_path)
    if not os.path.isfile(aggregated_json_path):
        logger.warning("GenBench console report skipped (missing %s)", aggregated_json_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import get_genbench_report as gbr
    except ImportError as e:
        logger.warning("Could not import get_genbench_report for console summary: %s", e)
        return
    try:
        pockets = gbr.load_pocket_dicts(aggregated_json_path)
        if not pockets:
            logger.warning("GenBench console report: empty aggregated JSON")
            return
        all_metrics, n_pockets = gbr.aggregate_metrics(pockets)
        logger.info(
            "GenBench3D console summary (equivalent to: python get_genbench_report.py %s)",
            aggregated_json_path,
        )
        gbr.print_report(all_metrics, n_pockets)
        gbr.log_genbench_two_line_metrics(logger, all_metrics, n_pockets)
    except Exception as e:
        logger.warning("GenBench console report failed: %s", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (typically training.yml).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.pt).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for sampling.')
    parser.add_argument('--num_proteins', type=int, default=5, help='Number of protein pockets to evaluate (from test split).')
    parser.add_argument(
        '--start_protein_idx',
        type=int,
        default=0,
        help='Start index in test split for protein pockets to evaluate.',
    )
    parser.add_argument('--num_ligands_per_protein', type=int, default=10, help='Number of ligands to sample per protein.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for sampling.')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale for conditioned models; 0.0 = unconditional (no CFG blend), >0 enables CFG.')
    parser.add_argument('--cfg_strategy', type=str, default='always', choices=['always', 'half_start', 'ramp_up'],
                        help='When to apply CFG during sampling. '
                             '"always" (default): apply CFG at every denoising step. '
                             '"half_start": skip CFG for the first (noisy) half of the schedule and '
                             '"ramp_up": ramp up CFG from 0 to 1 over the schedule, starting from the first step.'
                             'only enable it once sigma drops below the midpoint, letting the model '
                             'first find rough structure unconditionally.')
    parser.add_argument('--tmp_root', type=str, default='./tmp_eval', help='Root directory to create temporary evaluation folders.')
    parser.add_argument(
        '--sample_only',
        action='store_true',
        help='Only run sampling; save result_*.pt under the temp directory and exit (no evaluate_diffusion, SDF export, or GenBench).',
    )
    parser.add_argument('--eval_step', type=int, default=-1, help='Which diffusion step to evaluate (passed to evaluate_diffusion.py).')
    parser.add_argument('--docking_mode', type=str, default='none', choices=['qvina', 'vina_score', 'vina_dock', 'none'], help='Docking mode passed to evaluate_diffusion.py.')
    parser.add_argument(
        '--protein_root',
        type=str,
        default=None,
        help='Root for docking in evaluate_diffusion (PDB/ligand paths). Default: config.data.test_path if set, else ./data/test_set.',
    )
    parser.add_argument(
        '--test_set_dir',
        type=str,
        default=None,
        help='Root for GenBench3D native ligand lookup (pocket subdirs with .sdf). Default: --test_set_dir if passed, else config.data.test_path, else --protein_root.',
    )
    parser.add_argument('--exhaustiveness', type=int, default=16, help='Docking exhaustiveness passed to evaluate_diffusion.py.')
    # Guidance arguments (single or multiple)
    parser.add_argument('--guide_checkpoint', type=str, default=None, help='Path to a single guidance model checkpoint.')
    parser.add_argument('--guide_checkpoints', type=str, nargs='*', default=None, help='Paths to additional guidance model checkpoints (e.g., QED, SA).')
    parser.add_argument('--guide_scale_cord', type=float, default=1.0, help='Scale for coordinate guidance.')
    parser.add_argument('--guide_scale_categ', type=float, default=0.01, help='Scale for categorical guidance.')
    parser.add_argument('--guide_kind', type=int, default=2, help='Prior kind to guide to (default=2 i.e., Kd).')
    parser.add_argument('--time_scheduler', type=str, default=None, help='Time scheduler to use, overriding config.')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of diffusion steps for sampling, overriding config.')
    parser.add_argument(
        '--no_veda_noise_injection',
        action='store_true',
        help='Disable VEDA sampling extra noise injection (pos/type/bond remasking). Default: injection enabled.',
    )
    parser.add_argument(
        '--veda_noise_injection_rate',
        type=float,
        default=0.4,
        help='VEDA noise injection: scale factor applied to sigma before remasking (default: 0.4).',
    )
    parser.add_argument(
        '--veda_noise_injection_high_threshold',
        type=float,
        default=3.0,
        help='VEDA noise injection: only when all batch sigmas are below this value (default: 3).',
    )
    parser.add_argument(
        '--veda_noise_injection_low_threshold',
        type=float,
        default=0.0,
        help='VEDA noise injection: only when all batch sigmas are above this value (default: 0).',
    )
    parser.add_argument('--sample_config', type=str, default=None, help='Path to a yaml that defines guide_models (checkpoint, weight, gradient_scale_cord, gradient_scale_categ). Overrides --guide_checkpoints and related args.')
    # Optional: grouped SDF export + GenBench3D (off by default)
    parser.add_argument(
        '--export_grouped_sdf',
        action='store_true',
        help='After evaluation, export metrics to grouped *_generated.sdf (requires RDKit; enables saving metrics).',
    )
    parser.add_argument(
        '--sdf_grouped_dir',
        type=str,
        default=None,
        help='Directory for grouped SDFs. Default: <tmp_dir>/sdfs_grouped when exporting or running GenBench from this run.',
    )
    parser.add_argument(
        '--run_genbench',
        action='store_true',
        help='Run GenBench3D on *_generated.sdf under --sdf_grouped_dir (export in the same run or use an existing directory).',
    )
    parser.add_argument(
        '--gb3d_dir',
        type=str,
        default=None,
        help='GenBench3D repository root (contains sb_benchmark_mols.py). Required with --run_genbench.',
    )
    parser.add_argument(
        '--genbench_python',
        type=str,
        default='python',
        help='Python executable for GenBench3D subprocess (use conda env python if needed).',
    )
    parser.add_argument(
        '--no_genbench_console_report',
        action='store_true',
        help='With --run_genbench, do not print get_genbench_report.py-style summary after aggregation.',
    )
    parser.add_argument(
        '--genbench_do_conf_analysis',
        action='store_true',
        help='With --run_genbench, pass --do_conf_analysis to GenBench3D (Validity3D, TFD metrics, MMFF strain). Slower.',
    )
    parser.add_argument(
        '--genbench_no_vina',
        action='store_true',
        help='With --run_genbench, pass --no_vina to GenBench3D (skip Vina/minimized Vina; needs patched sb_benchmark_mols.py).',
    )
    args = parser.parse_args()

    if args.run_genbench and not args.gb3d_dir:
        parser.error('--run_genbench requires --gb3d_dir')
    if args.sample_only and (args.export_grouped_sdf or args.run_genbench):
        parser.error('--sample_only cannot be used with --export_grouped_sdf or --run_genbench (evaluation is skipped).')

    # Load global config (data split, seeds, etc.)
    config = misc.load_config(args.config)
    misc.seed_all(config.train.seed)

    # Resolve paths: config.data.test_path (e.g. ./data/test_set) is the canonical benchmark pocket tree.
    cfg_test_path = getattr(config.data, 'test_path', None)
    protein_root = args.protein_root if args.protein_root is not None else (cfg_test_path or './data/test_set')
    protein_root = os.path.abspath(protein_root)

    def resolve_genbench_test_set_dir():
        if args.test_set_dir is not None:
            return os.path.abspath(args.test_set_dir)
        if cfg_test_path:
            return os.path.abspath(cfg_test_path)
        return protein_root

    # Temporary directory for results
    os.makedirs(args.tmp_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    tmp_dir = tempfile.mkdtemp(prefix=f"quick_eval_{timestamp}_", dir=args.tmp_root)

    logger = misc.get_logger("quick_eval", log_dir=None)
    logger.info(f"Temporary evaluation directory: {tmp_dir}")
    logger.info("Docking protein_root (evaluate_diffusion): %s", protein_root)

    # Load the score (generation) model checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    ckpt_cfg = ckpt['config']

    # Build transforms – keep ligand atom mode consistent with the checkpoint
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt_cfg.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])

    # Load dataset (test split) using the transforms above
    logger.info("Loading dataset (test split)...")
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    test_set = subsets["test"]
    logger.info(f"Test set size: {len(test_set)}")

    if args.start_protein_idx < 0:
        parser.error('--start_protein_idx must be >= 0')
    if args.start_protein_idx >= len(test_set):
        parser.error(
            f'--start_protein_idx ({args.start_protein_idx}) must be smaller than test set size ({len(test_set)})'
        )
    end_protein_idx = min(args.start_protein_idx + args.num_proteins, len(test_set))
    protein_indices = range(args.start_protein_idx, end_protein_idx)
    num_proteins = len(protein_indices)
    logger.info(
        f"Will evaluate proteins in test index range [{args.start_protein_idx}, {end_protein_idx}) "
        f"(count={num_proteins}), {args.num_ligands_per_protein} ligands per protein."
    )

    # Initialise the generation model
    model = ScorePosNet3D(
        ckpt_cfg.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    # Apply CLI overrides if provided
    if args.time_scheduler is not None:
        model.config.time_scheduler = args.time_scheduler
    elif hasattr(config, "model") and hasattr(config.model, "time_scheduler"):
        model.config.time_scheduler = config.model.time_scheduler
    if hasattr(config, "model") and hasattr(config.model, "rho"):
        model.rho = config.model.rho
    if args.num_steps is not None:
        model.num_timesteps = args.num_steps
    elif hasattr(config, "model") and hasattr(config.model, "num_diffusion_timesteps"):
        model.num_timesteps = config.model.num_diffusion_timesteps
    if hasattr(config, "model") and hasattr(config.model, "dfm_type"):
        model.config.dfm_type = config.model.dfm_type
    if hasattr(config, "model") and hasattr(config.model, "veda_x_pred_mode"):
        model.config.veda_x_pred_mode = config.model.veda_x_pred_mode
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    logger.info(f"Loaded model from checkpoint: {args.checkpoint}")
    veda_noise_kwargs = {
        "noise_injection": not args.no_veda_noise_injection,
        "noise_injection_rate": args.veda_noise_injection_rate,
        "noise_injection_high_threshold": args.veda_noise_injection_high_threshold,
        "noise_injection_low_threshold": args.veda_noise_injection_low_threshold,
    }
    logger.info("VEDA noise injection: %s", veda_noise_kwargs)
    use_condition = getattr(ckpt_cfg.model, "use_condition", False)
    if use_condition:
        logger.info(
            f"Model uses conditioning (use_condition=True); sampling with cfg_scale={args.cfg_scale} "
            f"and target bins set to top tier (de novo, same as train quick eval)."
        )

    # ---------------------------------------------------------------------
    # Load guidance models (single or multiple)
    # ---------------------------------------------------------------------
    guide_models = []
    guide_configs = []

    if args.sample_config is not None:
        # Load yaml config defining guide models
        with open(args.sample_config, 'r') as f:
            sample_cfg = yaml.safe_load(f)
        guide_paths = [g['checkpoint'] for g in sample_cfg.get('guide_models', [])]
        guide_weights = [g.get('weight', 1.0) for g in sample_cfg.get('guide_models', [])]
        guide_cord = [g.get('gradient_scale_cord', args.guide_scale_cord) for g in sample_cfg.get('guide_models', [])]
        guide_categ = [g.get('gradient_scale_categ', args.guide_scale_categ) for g in sample_cfg.get('guide_models', [])]
    else:
        guide_paths = []
        if args.guide_checkpoint:
            guide_paths.append(args.guide_checkpoint)
        if args.guide_checkpoints:
            guide_paths.extend(args.guide_checkpoints)
        guide_weights = [1.0] * len(guide_paths)
        guide_cord = [args.guide_scale_cord] * len(guide_paths)
        guide_categ = [args.guide_scale_categ] * len(guide_paths)

    if guide_paths:
        logger.info(f'Loading {len(guide_paths)} guidance models...')
        for gp, w, cord, categ in zip(guide_paths, guide_weights, guide_cord, guide_categ):
            guide_ckpt = torch.load(gp, map_location=args.device)
            guide_model = DockGuideNet3D(
                guide_ckpt['config'].model,
                protein_atom_feature_dim=protein_featurizer.feature_dim,
                ligand_atom_feature_dim=ligand_featurizer.feature_dim,
            ).to(args.device)
            guide_model.load_state_dict(guide_ckpt['model'])
            guide_model.eval()
            guide_models.append(guide_model)
            guide_configs.append({
                'weight': w,
                'gradient_scale_cord': cord,
                'gradient_scale_categ': categ,
                'clamp_pred_min': None,
                'clamp_pred_max': None,
            })
    else:
        guide_models = None
        guide_configs = None

    # ---------------------------------------------------------------------
    # Sampling loop per protein
    # ---------------------------------------------------------------------
    sampling_sec_per_protein = []
    for data_id in protein_indices:
        data = test_set[data_id]
        logger.info(f"Sampling for protein index {data_id}...")

        if use_condition:
            num_bins = getattr(ckpt_cfg.model, "condition_bins", 5)
            best_bin = num_bins - 1
            data.vina_bin = torch.tensor(best_bin, dtype=torch.long)
            data.qed_bin = torch.tensor(best_bin, dtype=torch.long)
            data.sa_bin = torch.tensor(best_bin, dtype=torch.long)

        t_sample_start = time.perf_counter()
        if guide_models is not None:
            # Multi‑guide diffusion
            (
                pred_pos,
                pred_v,
                pred_pos_traj,
                pred_v_traj,
                pred_v0_traj,
                pred_vt_traj,
                pred_pos0_traj,
                time_list,
            ) = sample_multi_guided_diffusion(
                model=model,
                guide_models=guide_models,
                guide_configs=guide_configs,
                data=data,
                num_samples=args.num_ligands_per_protein,
                batch_size=args.batch_size,
                device=args.device,
                num_steps=model.num_timesteps,
                pos_only=False,
                center_pos_mode=ckpt_cfg.model.center_pos_mode,
                sample_num_atoms='prior',
                **veda_noise_kwargs,
            )
        else:
            # Fallback to original single‑guide diffusion
            (
                pred_pos,
                pred_v,
                pred_pos_traj,
                pred_v_traj,
                pred_v0_traj,
                pred_vt_traj,
                pred_pos0_traj,
                time_list,
            ) = sample_diffusion_ligand(
                model,
                data,
                args.num_ligands_per_protein,
                batch_size=args.batch_size,
                device=args.device,
                num_steps=model.num_timesteps,
                pos_only=False,
                center_pos_mode=ckpt_cfg.model.center_pos_mode,
                sample_num_atoms='prior',
                cfg_scale=args.cfg_scale,
                cfg_strategy=args.cfg_strategy,
                **veda_noise_kwargs,
            )
        sample_dt = time.perf_counter() - t_sample_start
        sampling_sec_per_protein.append((data_id, sample_dt))
        n_lig = max(args.num_ligands_per_protein, 1)
        logger.info(
            "Sampling time (protein index %d): %.4f s total, %.4f s per ligand (wall clock)",
            data_id,
            sample_dt,
            sample_dt / n_lig,
        )

        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_ligand_pos0_traj': pred_pos0_traj,
            'pred_ligand_v0_traj': pred_v0_traj,
            'pred_ligand_vt_traj': pred_vt_traj,
            'time': time_list,
        }
        if args.docking_mode != "none":
            result["ligand_filename"] = getattr(data, "ligand_filename", None)
            result["protein_filename"] = getattr(data, "protein_filename", None)

        out_path = os.path.join(tmp_dir, f"result_{data_id}.pt")
        torch.save(result, out_path)
        logger.info(f"Saved sampling result to: {out_path}")

    if sampling_sec_per_protein:
        dts = [dt for _, dt in sampling_sec_per_protein]
        total_s = sum(dts)
        n_p = len(sampling_sec_per_protein)
        n_lig_total = n_p * max(args.num_ligands_per_protein, 1)
        mean_per_p = total_s / n_p
        timing_payload = {
            "seconds_per_protein": [
                {"data_id": int(did), "seconds": float(dt)} for did, dt in sampling_sec_per_protein
            ],
            "summary": {
                "total_seconds": float(total_s),
                "num_proteins": n_p,
                "num_ligands_per_protein": args.num_ligands_per_protein,
                "total_ligands": n_lig_total,
                "mean_seconds_per_protein": float(mean_per_p),
                "min_seconds_per_protein": float(min(dts)),
                "max_seconds_per_protein": float(max(dts)),
                "mean_seconds_per_ligand": float(total_s / n_lig_total),
            },
        }
        timing_path = os.path.join(tmp_dir, "sampling_timing.json")
        with open(timing_path, "w") as tf:
            json.dump(timing_payload, tf, indent=2)
        logger.info(
            "Sampling timing: total=%.4f s | proteins=%d | mean per protein=%.4f s | min=%.4f s | max=%.4f s | "
            "mean per ligand=%.4f s | wrote %s",
            total_s,
            n_p,
            mean_per_p,
            min(dts),
            max(dts),
            total_s / n_lig_total,
            timing_path,
        )

    if args.sample_only:
        logger.info("Sample-only run finished; skipped evaluate_diffusion. Output directory: %s", tmp_dir)
        return

    # ---------------------------------------------------------------------
    # Run evaluation script on the generated samples
    # ---------------------------------------------------------------------
    sdf_grouped_dir = args.sdf_grouped_dir
    if sdf_grouped_dir is None and (args.export_grouped_sdf or args.run_genbench):
        sdf_grouped_dir = os.path.join(tmp_dir, "sdfs_grouped")
    save_metrics = bool(args.export_grouped_sdf)
    tagmol_names_tsv, tagmol_vals_tsv = None, None

    logger.info("Running evaluation on generated samples (in-process evaluate_diffusion)...")
    eval_results_dir = os.path.join(tmp_dir, "eval_results")
    os.makedirs(eval_results_dir, exist_ok=True)
    eval_logger = misc.get_logger("evaluate", log_dir=eval_results_dir)
    out, results = run_evaluation(
        sample_path=tmp_dir,
        eval_step=args.eval_step,
        eval_num_examples=num_proteins,
        docking_mode=args.docking_mode,
        protein_root=protein_root,
        verbose=args.docking_mode != "none",
        save=save_metrics,
        exhaustiveness=args.exhaustiveness,
        logger=eval_logger,
    )
    tagmol_names_tsv, tagmol_vals_tsv = report_evaluation_to_logger(
        out, results, args.docking_mode, eval_logger, one_line=True
    )

    if args.export_grouped_sdf:
        metrics_pt = os.path.join(tmp_dir, "eval_results", "metrics_%d.pt" % args.eval_step)
        out_sdf = sdf_grouped_dir or os.path.join(tmp_dir, "sdfs_grouped")
        export_metrics_to_grouped_sdf(metrics_pt, out_sdf, logger)

    if args.run_genbench:
        gdir = sdf_grouped_dir or os.path.join(tmp_dir, "sdfs_grouped")
        genbench_test_set = resolve_genbench_test_set_dir()
        logger.info("GenBench3D native ligand root: %s", genbench_test_set)
        run_genbench3d_suite(
            gdir,
            os.path.abspath(args.gb3d_dir),
            genbench_test_set,
            args.genbench_python,
            logger,
            do_conf_analysis=args.genbench_do_conf_analysis,
            no_vina=args.genbench_no_vina,
        )
        if not args.no_genbench_console_report:
            agg_json = os.path.join(os.path.abspath(gdir), "genbench_results", "all_results_aggregated.json")
            print_genbench_console_report(agg_json, logger)

    if tagmol_names_tsv is not None and tagmol_vals_tsv is not None:
        logger.info("=" * 60)
        logger.info("TAGMol metrics (repeat, tab-separated)")
        logger.info("METRICS_ONE_LINE_HEAD\t%s", tagmol_names_tsv)
        logger.info("METRICS_ONE_LINE_VAL\t%s", tagmol_vals_tsv)

    logger.info("Quick evaluation finished.")


if __name__ == "__main__":
    main()

