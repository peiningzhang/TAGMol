#!/usr/bin/env python3
"""Build MolPilot-style ``torch.save`` dict with ``all_results`` from PocketXMol sbdd_csd flat SDFs.

Rows match what ``summarize_molpilot_pt_like_evaluate_diffusion.py`` expects:
``mol``, ``chem_results`` (qed/sa from ``mol_metric.csv`` or RDKit), ``ligand_filename`` (pocket id
for per-pocket diversity), optional ``vina`` if present later.

Typical layout::

    <run_dir>/gen_info.csv
    <run_dir>/mol_metric.csv   # first column = SDF filename
    <run_dir>/SDF/*.sdf
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.evaluation import scoring_func


def _load_mol_metrics(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("") or "").strip()
            if not fn:
                for _k, v in row.items():
                    if isinstance(v, str) and v.endswith(".sdf"):
                        fn = v.strip()
                        break
            if not fn:
                continue
            rec = {}
            for k, v in row.items():
                if k == "" or v is None or str(v).strip() == "":
                    continue
                try:
                    rec[k] = float(v)
                except (ValueError, TypeError):
                    rec[k] = v
            out[fn] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="PocketXMol run directory")
    ap.add_argument(
        "--out-pt",
        required=True,
        help="Output .pt path (dict with all_results=list[dict])",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    sdf_root = os.path.join(run_dir, "SDF")
    gen_info = os.path.join(run_dir, "gen_info.csv")
    mol_metric_path = os.path.join(run_dir, "mol_metric.csv")

    if not os.path.isdir(sdf_root):
        raise SystemExit("Missing SDF dir: %s" % sdf_root)
    if not os.path.isfile(gen_info):
        raise SystemExit("Missing gen_info.csv: %s" % gen_info)

    metrics_by_fn = _load_mol_metrics(mol_metric_path)
    all_results = []

    with open(gen_info, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            pid = (row.get("data_id") or "").strip()
            if not fn or not pid:
                continue
            src = os.path.join(sdf_root, fn)
            if not os.path.isfile(src):
                continue
            suppl = Chem.SDMolSupplier(src, removeHs=False, sanitize=False)
            mol = suppl[0] if suppl and len(suppl) else None
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            except Exception:
                pass
            mm = metrics_by_fn.get(fn, {})
            if "qed" in mm and "sa" in mm:
                chem = {"qed": float(mm["qed"]), "sa": float(mm["sa"])}
            else:
                try:
                    chem = scoring_func.get_chem(mol)
                except Exception:
                    chem = {"qed": 0.0, "sa": 10.0}

            all_results.append(
                {
                    "mol": mol,
                    "chem_results": chem,
                    "ligand_filename": "%s.sdf" % pid,
                    "vina": None,
                }
            )

    out_pt = os.path.abspath(os.path.expanduser(args.out_pt))
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save({"all_results": all_results}, out_pt)
    print("Saved %d rows to %s" % (len(all_results), out_pt))


if __name__ == "__main__":
    main()
