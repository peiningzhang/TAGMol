#!/usr/bin/env python3
"""Group PocketXMol sbdd_csd flat SDFs (0.sdf, 1.sdf, …) into GenBench-style *_generated.sdf.

Expects ``gen_info.csv`` in the run directory (columns include ``data_id``, ``filename``) and
molecules under ``<run_dir>/SDF/<filename>``.

NOTE: Keys are ``csd_*`` (CSD pockets), not CrossDocked names under TAGMol ``data/test_set``.
``run_genbench_eval.py`` will not find native ligands unless you provide a matching test tree.
This script still produces the same *file layout* as MolPilot / TAGMol grouped exports.
"""
from __future__ import annotations

import argparse
import csv
import os
import re

from rdkit import Chem


def _sanitize_for_export(mol):
    """Avoid kekulization failures on SDWriter (same pattern as utils/reconstruct.py)."""
    try:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        pass


def _natural_sdf_key(name: str) -> tuple:
    """Sort filenames so 2.sdf < 10.sdf; keep *-bad.sdf last within same numeric stem."""
    base = name.replace(".sdf", "")
    m = re.match(r"^(\d+)(-bad)?$", base)
    if m:
        return (int(m.group(1)), m.group(2) or "")
    return (10**9, base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="PocketXMol run dir containing gen_info.csv and SDF/",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for <data_id>_generated.sdf",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    sdf_root = os.path.join(run_dir, "SDF")
    gen_info = os.path.join(run_dir, "gen_info.csv")

    if not os.path.isfile(gen_info):
        raise SystemExit("Missing gen_info.csv: %s" % gen_info)
    os.makedirs(out_dir, exist_ok=True)

    by_pocket: dict[str, list[str]] = {}
    with open(gen_info, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("data_id") or "").strip()
            fn = (row.get("filename") or "").strip()
            if not pid or not fn:
                continue
            by_pocket.setdefault(pid, []).append(fn)

    n_files = 0
    n_mols = 0
    for pocket_id in sorted(by_pocket.keys()):
        names = sorted(set(by_pocket[pocket_id]), key=_natural_sdf_key)
        out_path = os.path.join(out_dir, "%s_generated.sdf" % pocket_id)
        w = Chem.SDWriter(out_path)
        try:
            w.SetKekulize(False)
        except Exception:
            pass
        pocket_n = 0
        for fn in names:
            src = os.path.join(sdf_root, fn)
            if not os.path.isfile(src):
                continue
            suppl = Chem.SDMolSupplier(src, removeHs=False, sanitize=False)
            mol = suppl[0] if suppl and len(suppl) else None
            if mol is None:
                continue
            _sanitize_for_export(mol)
            mol.SetProp("_pxm_filename", fn)
            mol.SetProp("_pxm_data_id", pocket_id)
            w.write(mol)
            pocket_n += 1
        w.close()
        if pocket_n == 0:
            try:
                os.remove(out_path)
            except OSError:
                pass
            continue
        n_files += 1
        n_mols += pocket_n

    print(
        "Wrote %d pockets, %d molecules under %s" % (n_files, n_mols, out_dir)
    )


if __name__ == "__main__":
    main()
