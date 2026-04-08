import argparse
import os
from collections import defaultdict

import torch
import rdkit
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(
        description="Export grouped SDFs from eval metrics_*.pt (same layout as evaluate_diffusion)."
    )
    parser.add_argument(
        "sample_path",
        type=str,
        help="Directory that contains eval_results/ (e.g. experiments/run_1 or ../AliDiff/sampling_results).",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=-1,
        help="Checkpoint step in the filename metrics_<step>.pt (default: -1).",
    )
    args = parser.parse_args()

    base = os.path.abspath(os.path.expanduser(args.sample_path.strip()))
    pt_file = os.path.join(base, "eval_results", f"metrics_{args.eval_step}.pt")
    out_dir = os.path.join(base, "sdfs_grouped")

    if not os.path.isfile(pt_file):
        raise SystemExit(f"Metrics file not found: {pt_file}")

    os.makedirs(out_dir, exist_ok=True)

    try:
        data = torch.load(pt_file, map_location="cpu")
    except RuntimeError as exc:
        msg = str(exc)
        if "ENDMOL" in msg or "pickle format" in msg.lower():
            raise SystemExit(
                "Cannot load this metrics file: RDKit could not unpickle embedded molecules.\n"
                f"  File: {pt_file}\n"
                f"  RDKit in this env: {getattr(rdkit, '__version__', 'unknown')}\n"
                "  The .pt was almost certainly saved with a newer RDKit (log showed pickle "
                "format 16.1 vs your 13.0).\n"
                "  Fix: upgrade RDKit in this conda env to match (or exceed) the version "
                "used when creating the file, e.g. conda install -c conda-forge rdkit, "
                "or run this script in the same environment as AliDiff.\n"
                f"  Underlying error: {exc}"
            ) from exc
        raise
    results = data.get("all_results", [])

    grouped_mols = defaultdict(list)

    for i, res in enumerate(results):
        mol = res.get("mol")
        if mol is None:
            continue

        target_name = res.get("ligand_filename", f"target_{i}")
        if isinstance(target_name, str):
            if target_name.startswith("LIGAND_"):
                target_name = target_name.replace("LIGAND_", "")

            target_name_base = target_name.split(".")[0].replace("/", "_")
        else:
            target_name_base = f"target_{i}"

        if "chem_results" in res and isinstance(res["chem_results"], dict):
            if "qed" in res["chem_results"]:
                mol.SetProp("QED", str(res["chem_results"]["qed"]))
            if "sa" in res["chem_results"]:
                mol.SetProp("SA", str(res["chem_results"]["sa"]))

        if "vina" in res and res["vina"]:
            try:
                val = res["vina"].get("score_only", [{}])[0].get("affinity", "")
                mol.SetProp("Vina", str(val))
            except Exception:
                pass

        grouped_mols[target_name_base].append(mol)

    count_files = 0
    count_mols = 0
    for target_name, mols in grouped_mols.items():
        sdf_path = os.path.join(out_dir, f"{target_name}_generated.sdf")
        writer = Chem.SDWriter(sdf_path)
        for m in mols:
            writer.write(m)
            count_mols += 1
        writer.close()
        count_files += 1

    print(
        f"Successfully exported {count_mols} molecules into {count_files} grouped SDF files in {out_dir}!"
    )


if __name__ == "__main__":
    main()
