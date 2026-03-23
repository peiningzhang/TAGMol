"""
Quick one-shot evaluation script for TAGMol / VEDA.

Usage:
    python scripts/quick_evaluate.py \
        --config configs/training.yml \
        --checkpoint logs_diffusion/.../checkpoints/176000.pt

Behavior:
    - Uses the provided config to load the dataset (test split).
    - Loads the model from the given checkpoint.
    - For the first N proteins (default 5), generates M ligands each (default 10).
    - Saves all sampling results into a temporary directory.
    - Runs `scripts/evaluate_diffusion.py` on that directory.

Notes:
    - Does NOT write anything into the main experiments directory.
    - All intermediate .pt files are stored under a temp folder inside ./tmp_eval/.
"""

import argparse
import os
import tempfile
import subprocess
from datetime import datetime

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from models.molopt_guide_model import DockGuideNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from scripts.sample_guided_diffusion import sample_guided_diffusion_ligand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (typically training.yml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for sampling.",
    )
    parser.add_argument(
        "--num_proteins",
        type=int,
        default=5,
        help="Number of protein pockets to evaluate (from test split).",
    )
    parser.add_argument(
        "--num_ligands_per_protein",
        type=int,
        default=10,
        help="Number of ligands to sample per protein.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--tmp_root",
        type=str,
        default="./tmp_eval",
        help="Root directory to create temporary evaluation folders.",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=-1,
        help="Which diffusion step to evaluate (passed to evaluate_diffusion.py).",
    )
    parser.add_argument(
        "--docking_mode",
        type=str,
        default="none",
        choices=["qvina", "vina_score", "vina_dock", "none"],
        help="Docking mode passed to evaluate_diffusion.py.",
    )
    parser.add_argument(
        "--protein_root",
        type=str,
        default="./data/test_set",
        help="Protein root for docking (dir containing PDB files). Use data/test_set if crossdocked path lacks pdbs.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=16,
        help="Docking exhaustiveness passed to evaluate_diffusion.py.",
    )
    # Guided diffusion params
    parser.add_argument(
        "--guide_checkpoint",
        type=str,
        default=None,
        help="Path to trained property guidance model checkpoint.",
    )
    parser.add_argument(
        "--guide_scale_cord", type=float, default=1.0, help="Scale for coordinate guidance."
    )
    parser.add_argument(
        "--guide_scale_categ", type=float, default=0.01, help="Scale for categorical guidance."
    )
    parser.add_argument(
        "--guide_kind", type=int, default=2, help="Prior kind to guide to (default=2 i.e., Kd)"
    )
    parser.add_argument(
        "--time_scheduler", type=str, default=None, help="Time scheduler to use, overriding config"
    )
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Number of diffusion steps for sampling, overriding config"
    )
    args = parser.parse_args()

    # Load config (for data + basic settings)
    config = misc.load_config(args.config)
    misc.seed_all(config.train.seed)

    # Prepare temporary directory
    os.makedirs(args.tmp_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    tmp_dir = tempfile.mkdtemp(prefix=f"quick_eval_{timestamp}_", dir=args.tmp_root)

    logger = misc.get_logger("quick_eval", log_dir=None)
    logger.info(f"Temporary evaluation directory: {tmp_dir}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    ckpt_config = ckpt["config"]

    # Transforms (use ligand_atom_mode from checkpoint config to stay consistent)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose(
        [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
    )

    # Dataset: use data section from main config (path/split), but transforms from above
    logger.info("Loading dataset (test split)...")
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    test_set = subsets["test"]
    logger.info(f"Test set size: {len(test_set)}")

    num_proteins = min(args.num_proteins, len(test_set))
    logger.info(
        f"Will evaluate on first {num_proteins} proteins, "
        f"{args.num_ligands_per_protein} ligands per protein."
    )

    # Model
    model = ScorePosNet3D(
        ckpt_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    # 在 main 里，model = ScorePosNet3D(...) 之后加：
    if getattr(args, "time_scheduler", None) is not None:
        model.config.time_scheduler = args.time_scheduler
    elif hasattr(config, "model") and hasattr(config.model, "time_scheduler"):
        model.config.time_scheduler = config.model.time_scheduler
    if hasattr(config, "model") and hasattr(config.model, "rho"):
        model.rho = config.model.rho
    if getattr(args, "num_steps", None) is not None:
        model.num_timesteps = args.num_steps
    elif hasattr(config, "model") and hasattr(config.model, "num_diffusion_timesteps"):
        model.num_timesteps = config.model.num_diffusion_timesteps
    if hasattr(config, "model") and hasattr(config.model, "dfm_type"):
        model.config.dfm_type = config.model.dfm_type
    if hasattr(config, "model") and hasattr(config.model, "veda_x_pred_mode"):
        model.config.veda_x_pred_mode = config.model.veda_x_pred_mode
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded model from checkpoint: {args.checkpoint}")

    # Optionally load guide model
    guide_model = None
    if args.guide_checkpoint is not None:
        logger.info(f"Loading guide model from {args.guide_checkpoint}...")
        guide_ckpt = torch.load(args.guide_checkpoint, map_location=args.device)
        guide_model = DockGuideNet3D(
            guide_ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        guide_model.load_state_dict(guide_ckpt['model'])
        guide_model.eval()

    # Sampling loop
    for data_id in range(num_proteins):
        data = test_set[data_id]
        logger.info(f"Sampling for protein index {data_id}...")

        if guide_model is not None:
            (
                pred_pos,
                pred_v,
                pred_pos_traj,
                pred_v_traj,
                pred_v0_traj,
                pred_vt_traj,
                pred_pos0_traj,
                time_list,
            ) = sample_guided_diffusion_ligand(
                model=model,
                guide_model=guide_model,
                data=data,
                num_samples=args.num_ligands_per_protein,
                kind=args.guide_kind,
                gradient_scale_cord=args.guide_scale_cord,
                gradient_scale_categ=args.guide_scale_categ,
                clamp_pred_min=None,
                clamp_pred_max=None,
                batch_size=args.batch_size,
                device=args.device,
                num_steps=model.num_timesteps,
                pos_only=False,
                center_pos_mode=ckpt_config.model.center_pos_mode,
                sample_num_atoms="prior",
            )
        else:
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
                center_pos_mode=ckpt_config.model.center_pos_mode,
                sample_num_atoms="prior",
            )

        result = {
            "data": data,
            "pred_ligand_pos": pred_pos,
            "pred_ligand_v": pred_v,
            "pred_ligand_pos_traj": pred_pos_traj,
            "pred_ligand_v_traj": pred_v_traj,
            "pred_ligand_pos0_traj": pred_pos0_traj,
            "pred_ligand_v0_traj": pred_v0_traj,
            "time": time_list,
        }
        if args.docking_mode != "none":
            result["ligand_filename"] = getattr(data, "ligand_filename", None)
            result["protein_filename"] = getattr(data, "protein_filename", None)

        out_path = os.path.join(tmp_dir, f"result_{data_id}.pt")
        torch.save(result, out_path)
        logger.info(f"Saved sampling result to: {out_path}")

    # Run evaluation script on the temporary directory
    logger.info("Running evaluation on generated samples...")
    eval_cmd = [
        "python",
        "scripts/evaluate_diffusion.py",
        tmp_dir,
        "--eval_step",
        str(args.eval_step),
        "--eval_num_examples",
        str(num_proteins),
        "--docking_mode",
        args.docking_mode,
        "--save",
        "False",
        "--one_line",
    ]
    if args.docking_mode != "none":
        eval_cmd += ["--verbose", "True"]
        eval_cmd += [
            "--protein_root",
            args.protein_root,
            "--exhaustiveness",
            str(args.exhaustiveness),
        ]
    logger.info(f"Eval command: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=False)

    logger.info("Quick evaluation finished.")


if __name__ == "__main__":
    main()

