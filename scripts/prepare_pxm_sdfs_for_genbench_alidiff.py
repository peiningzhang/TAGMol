#!/usr/bin/env python3
"""Remap PocketXMol grouped ``csd_NNNNNN_generated.sdf`` to GenBench-style names from AliDiff split.

``run_genbench_eval.py`` matches each ``<key>_generated.sdf`` to
``test_set/<pocket_dir>/<ligand>.sdf`` via ``<pocket_dir>_<ligand_stem>``.

AliDiff ``split_by_name.pt`` lists ``test`` as ``(protein_rel, ligand_rel)`` pairs in the same
order as ``csd_100000 + i`` in PocketXMol ``gen_info.csv`` / grouped exports.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-by-name",
        required=True,
        help="AliDiff split_by_name.pt (must contain key 'test')",
    )
    ap.add_argument(
        "--pxm-grouped-dir",
        required=True,
        help="Directory with csd_100000_generated.sdf, ... (TAGMol grouped export)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for <pocket>_<lig>_generated.sdf copies",
    )
    args = ap.parse_args()

    split_path = os.path.abspath(args.split_by_name)
    src_dir = os.path.abspath(args.pxm_grouped_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    obj = torch.load(split_path, map_location="cpu")
    if "test" not in obj:
        raise SystemExit("split file must contain 'test' key")
    split_test = obj["test"]
    n = len(split_test)

    for i, (protein_rel, ligand_rel) in enumerate(split_test):
        pocket = os.path.dirname(ligand_rel)
        lig_base = os.path.splitext(os.path.basename(ligand_rel))[0]
        key = "%s_%s" % (pocket, lig_base)
        csd_id = 100000 + i
        src = os.path.join(src_dir, "csd_%d_generated.sdf" % csd_id)
        dst = os.path.join(out_dir, "%s_generated.sdf" % key)
        if not os.path.isfile(src):
            raise SystemExit("missing grouped SDF: %s" % src)
        shutil.copy2(src, dst)
    print("Copied %d files to %s" % (n, out_dir))


if __name__ == "__main__":
    main()
