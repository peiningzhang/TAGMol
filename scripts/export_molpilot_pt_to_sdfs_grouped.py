#!/usr/bin/env python3
"""Export MolPilot-style torch.save() results to GenBench-compatible *_generated.sdf files.

Input may be a flat list[dict] or dict with key ``all_results`` (same per-row schema as
``export_to_sdf_grouped.py`` expects). Writes one SDF per pocket (grouped by ligand_filename).
"""
import argparse
import os
from collections import defaultdict

import torch
import rdkit
from rdkit import Chem


def _normalize_results(raw):
    if isinstance(raw, dict):
        if "all_results" in raw:
            return raw["all_results"]
        raise SystemExit(
            "Expected a list or a dict with 'all_results'; got dict keys: %s"
            % (list(raw.keys())[:20],)
        )
    if isinstance(raw, list):
        return raw
    raise SystemExit("Unsupported top-level type: %s" % type(raw).__name__)


def export_grouped_sdfs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
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
            # ResGen / SeFMol Zenodo baselines use names ending in _gen; GenBench test_set keys
            # omit this suffix, so one trailing _gen is dropped for file matching.
            if target_name_base.endswith("_gen"):
                target_name_base = target_name_base[: -len("_gen")]
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
            except Exception:
                pass

        grouped_mols[target_name_base].append(mol)

    count_files = 0
    count_mols = 0
    for target_name, mols in grouped_mols.items():
        sdf_path = os.path.join(out_dir, "%s_generated.sdf" % target_name)
        w = Chem.SDWriter(sdf_path)
        for m in mols:
            w.write(m)
            count_mols += 1
        w.close()
        count_files += 1

    return count_files, count_mols


def main():
    p = argparse.ArgumentParser(
        description="Export a MolPilot / flat-metrics .pt file to grouped *_generated.sdf for GenBench3D."
    )
    p.add_argument(
        "--pt",
        required=True,
        help="Path to .pt (list[dict] or dict with all_results).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write *_generated.sdf (e.g. benchmarks/from_molpilot/molcraft_sdfs_grouped).",
    )
    args = p.parse_args()

    pt_path = os.path.abspath(os.path.expanduser(args.pt))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    if not os.path.isfile(pt_path):
        raise SystemExit("Input not found: %s" % pt_path)

    try:
        raw = torch.load(pt_path, map_location="cpu")
    except RuntimeError as exc:
        msg = str(exc)
        if "ENDMOL" in msg or "pickle format" in msg.lower():
            raise SystemExit(
                "Cannot load molecules (RDKit unpickle failed).\n"
                "  File: %s\n"
                "  RDKit in this env: %s\n"
                "  Upgrade RDKit to match the version used when the .pt was written, "
                "then retry.\n"
                "  Underlying error: %s" % (pt_path, getattr(rdkit, "__version__", "?"), exc)
            ) from exc
        raise

    results = _normalize_results(raw)
    n_files, n_mols = export_grouped_sdfs(results, out_dir)
    print(
        "Exported %d molecules into %d grouped SDF files under %s"
        % (n_mols, n_files, out_dir)
    )


if __name__ == "__main__":
    main()
