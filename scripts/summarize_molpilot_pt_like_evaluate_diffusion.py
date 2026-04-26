#!/usr/bin/env python3
"""
Aggregate MolPilot-style flat *.pt (list[dict] with mol, vina, chem_results, …) using the same
metric definitions as scripts/evaluate_diffusion.run_evaluation where possible.

MolPilot rows usually lack pred_pos/pred_v; stability and pair distances are taken from the
stored RDKit mol conformer (geometrically aligned with TAGMol's post-reconstruct 3D used in eval).

Writes a one-row TSV (header + values) suitable for Excel, defaulting next to the input .pt.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.evaluation import analyze, eval_atom_type, eval_bond_length, scoring_func
from utils.evaluation.similarity import mean_pairwise_tanimoto

RDLogger.DisableLog("rdapp.*")

JSD_BOND_COLS = [
    "JSD_6-6|1",
    "JSD_6-6|2",
    "JSD_6-6|4",
    "JSD_6-7|1",
    "JSD_6-7|2",
    "JSD_6-7|4",
    "JSD_6-8|1",
    "JSD_6-8|2",
    "JSD_6-8|4",
    "JSD_All_12A",
    "JSD_CC_2A",
]

OUTPUT_COLS = [
    "mol_stable",
    "atm_stable",
    "recon_success",
    "eval_success",
    "complete",
] + JSD_BOND_COLS + [
    "atom_type_js",
    "n_recon",
    "n_complete",
    "n_eval",
    "QED_mean",
    "QED_med",
    "SA_mean",
    "SA_med",
    "Diversity_mean",
    "Diversity_median",
    "Vina_score_mean",
    "Vina_score_median",
    "Vina_min_mean",
    "Vina_min_median",
    "Dock_mean",
    "Dock_median",
]


def _normalize_results(raw):
    if isinstance(raw, dict):
        if "all_results" in raw:
            return raw["all_results"]
        raise SystemExit("dict input must contain 'all_results'")
    if isinstance(raw, list):
        return raw
    raise SystemExit("unsupported top-level type: %s" % type(raw).__name__)


def _mol_positions_elements(mol):
    if mol.GetNumConformers() == 0:
        return None, None
    conf = mol.GetConformer()
    pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    elements = np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, elements


def _vina_affinity(vina, branch, mode="first"):
    if not vina or branch not in vina:
        return None
    arr = vina[branch]
    if not arr:
        return None
    entry = arr[0]
    if isinstance(entry, dict) and "affinity" in entry:
        return entry["affinity"]
    return None


def aggregate_like_evaluate(rows):
    num_samples = len(rows)
    all_mol_stable = all_atom_stable = all_n_atom = 0
    n_recon_success = n_eval_success = n_complete = 0
    all_bond_dist = []
    success_pair_dist = []
    success_atom_types = Counter()
    results = []

    for row in rows:
        mol = row.get("mol")
        if mol is not None:
            n_recon_success += 1
            try:
                pos, elements = _mol_positions_elements(mol)
                if pos is not None and len(pos) == len(elements):
                    st = analyze.check_stability(pos, elements)
                    all_mol_stable += int(st[0])
                    all_atom_stable += int(st[1])
                    all_n_atom += int(st[2])
            except Exception:
                pass

        smi = row.get("smiles")
        if smi is None and mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = None
        if mol is not None and smi and "." not in smi:
            n_complete += 1

        vina = row.get("vina")
        if vina is None or mol is None:
            continue
        n_eval_success += 1
        try:
            chem = row.get("chem_results")
            if not isinstance(chem, dict) or "qed" not in chem:
                chem = scoring_func.get_chem(mol)
            pos, elements = _mol_positions_elements(mol)
            if pos is None:
                continue
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pos, elements)
            success_pair_dist.extend(pair_dist)
            success_atom_types.update(Counter(int(e) for e in elements))
            all_bond_dist.extend(eval_bond_length.bond_distance_from_mol(mol))
            results.append(
                {
                    "mol": mol,
                    "chem_results": chem,
                    "vina": vina,
                    "ligand_filename": row.get("ligand_filename"),
                    "protein_filename": row.get("protein_filename"),
                }
            )
        except Exception:
            continue

    fraction_mol_stable = all_mol_stable / num_samples if num_samples else 0.0
    fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom > 0 else 0.0
    fraction_recon = n_recon_success / num_samples if num_samples else 0.0
    fraction_eval = n_eval_success / num_samples if num_samples else 0.0
    fraction_complete = n_complete / num_samples if num_samples else 0.0

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    if success_pair_dist:
        success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
        success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    else:
        success_js_metrics = {}
    atom_type_js = (
        eval_atom_type.eval_atom_type_distribution(success_atom_types)
        if sum(success_atom_types.values()) > 0
        else None
    )

    qed = [r["chem_results"]["qed"] for r in results if r.get("chem_results")]
    sa = [r["chem_results"]["sa"] for r in results if r.get("chem_results")]
    out = {
        "mol_stable": fraction_mol_stable,
        "atm_stable": fraction_atm_stable,
        "recon_success": fraction_recon,
        "eval_success": fraction_eval,
        "complete": fraction_complete,
        "n_recon": n_recon_success,
        "n_complete": n_complete,
        "n_eval": len(results),
        "n_samples": num_samples,
        "atom_type_js": atom_type_js,
        "QED_mean": float(np.mean(qed)) if qed else None,
        "QED_med": float(np.median(qed)) if qed else None,
        "SA_mean": float(np.mean(sa)) if sa else None,
        "SA_med": float(np.median(sa)) if sa else None,
    }
    for k, v in c_bond_length_dict.items():
        out[k] = v
    for k, v in success_js_metrics.items():
        out[k] = v

    pocket_key_to_mols = defaultdict(list)
    for r in results:
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
        out["Diversity_mean"] = float(np.mean(pocket_divs))
        out["Diversity_median"] = float(np.median(pocket_divs))
    else:
        out["Diversity_mean"] = None
        out["Diversity_median"] = None

    docks = []
    scores = []
    mins = []
    for r in results:
        v = r["vina"]
        daf = _vina_affinity(v, "dock")
        if daf is not None:
            docks.append(float(daf))
        so = _vina_affinity(v, "score_only")
        if so is not None:
            scores.append(float(so))
        mn = _vina_affinity(v, "minimize")
        if mn is not None:
            mins.append(float(mn))
    out["Dock_mean"] = float(np.mean(docks)) if docks else None
    out["Dock_median"] = float(np.median(docks)) if docks else None
    out["Vina_score_mean"] = float(np.mean(scores)) if scores else None
    out["Vina_score_median"] = float(np.median(scores)) if scores else None
    out["Vina_min_mean"] = float(np.mean(mins)) if mins else None
    out["Vina_min_median"] = float(np.median(mins)) if mins else None

    return out


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return "%.6g" % v
    return str(v)


def row_tsv(out):
    parts = []
    for name in OUTPUT_COLS:
        if name in ("JSD_All_12A", "JSD_CC_2A"):
            key = name
        elif name.startswith("JSD_"):
            key = name
        else:
            key = name
        parts.append(_fmt(out.get(key)))
    return "\t".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True, help="Path to MolPilot-style .pt")
    ap.add_argument(
        "--out",
        default=None,
        help="Output TSV path (default: same dir as .pt, stem + _evaluate_like.tsv)",
    )
    args = ap.parse_args()
    pt_path = os.path.abspath(args.pt)
    if not os.path.isfile(pt_path):
        raise SystemExit("not found: %s" % pt_path)

    raw = torch.load(pt_path, map_location="cpu")
    rows = _normalize_results(raw)
    out = aggregate_like_evaluate(rows)

    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(pt_path)
        out_path = base + "_evaluate_like.tsv"

    header = "\t".join(OUTPUT_COLS)
    line = row_tsv(out)
    with open(out_path, "w") as f:
        f.write(header + "\n")
        f.write(line + "\n")
    print("Wrote %s" % out_path)
    print(header)
    print(line)


if __name__ == "__main__":
    main()
