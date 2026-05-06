#!/usr/bin/env python3
"""
Merge evaluate_diffusion shard outputs (eval_shards/shard_*/*/eval_results/metrics_*.pt + log.txt)
into one TSV row. Pair-distance JSDs (JSD_All_12A, JSD_CC_2A) are mean-of-shard approximations unless
full merged pair profiles are recomputed from raw result_*.pt.

GenBench3D / 2D columns are empty unless --genbench-json points to aggregated GenBench JSON/list.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict

import numpy as np
import torch

from utils.evaluation import eval_atom_type, eval_bond_length
from utils.evaluation.similarity import mean_pairwise_tanimoto


def _parse_metrics_one_line(log_path: str) -> tuple[list[str], list[str]] | tuple[None, None]:
    """Return (headers, value_strings) from METRICS_ONE_LINE_* in log."""
    head, vals = None, None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "METRICS_ONE_LINE_HEAD" in line:
                part = line.split("METRICS_ONE_LINE_HEAD", 1)[1].strip()
                head = [x.strip() for x in re.split(r"\t+", part) if x.strip()]
            elif "METRICS_ONE_LINE_VAL" in line:
                part = line.split("METRICS_ONE_LINE_VAL", 1)[1].strip()
                vals = [x.strip() for x in re.split(r"\t+", part) if x.strip()]
    if not head or not vals or len(head) != len(vals):
        return None, None
    return head, vals


def _mean_shard_metric(shard_logs: list[str], key: str) -> float | None:
    xs = []
    for lp in shard_logs:
        h, v = _parse_metrics_one_line(lp)
        if not h:
            continue
        try:
            i = h.index(key)
            xs.append(float(v[i]))
        except (ValueError, IndexError):
            continue
    if not xs:
        return None
    return float(statistics.mean(xs))


def merge_metrics(sample_dir: str, eval_step: int = -1) -> dict:
    """Build flat dict of merged metrics from shard PT files and logs."""
    shard_roots = sorted(glob.glob(os.path.join(sample_dir, "eval_shards", "shard_*")))
    if not shard_roots:
        raise FileNotFoundError("No eval_shards/shard_* under %s" % sample_dir)

    pt_paths = sorted(
        glob.glob(os.path.join(sample_dir, "eval_shards", "shard_*", "eval_results", "metrics_%s.pt" % eval_step))
    )
    log_paths = sorted(glob.glob(os.path.join(sample_dir, "eval_shards", "shard_*", "eval_results", "log.txt")))

    all_results = []
    merged_bond_dist = []
    stab_accum = {k: [] for k in ["mol_stable", "atm_stable", "recon_success", "eval_success", "complete"]}

    for p in pt_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        all_results.extend(d["all_results"])
        merged_bond_dist.extend(d["bond_length"])
        st = d["stability"]
        for k in stab_accum:
            stab_accum[k].append(st[k])

    for k in stab_accum:
        stab_accum[k] = float(np.mean(stab_accum[k]))

    c_prof = eval_bond_length.get_bond_length_profile(merged_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_prof)

    c_atom = Counter()
    for r in all_results:
        mol = r["mol"]
        for a in mol.GetAtoms():
            c_atom[a.GetAtomicNum()] += 1
    atom_type_js = eval_atom_type.eval_atom_type_distribution(c_atom) if sum(c_atom.values()) > 0 else None

    qed = [r["chem_results"]["qed"] for r in all_results]
    sa = [r["chem_results"]["sa"] for r in all_results]

    pocket_key_to_mols = defaultdict(list)
    for r in all_results:
        pk = r.get("protein_filename") or r.get("ligand_filename") or "unknown"
        pocket_key_to_mols[pk].append(r["mol"])
    pocket_mean_sims = []
    pocket_divs = []
    for mols in pocket_key_to_mols.values():
        mp = mean_pairwise_tanimoto(mols)
        if mp is not None:
            pocket_mean_sims.append(mp)
            pocket_divs.append(1.0 - mp)

    Mean_pairwise_Tanimoto = float(np.mean(pocket_mean_sims)) if pocket_mean_sims else None
    Diversity_mean = float(np.mean(pocket_divs)) if pocket_divs else None
    Diversity_median = float(np.median(pocket_divs)) if pocket_divs else None

    vina_score_only = [r["vina"]["score_only"][0]["affinity"] for r in all_results]
    vina_min = [r["vina"]["minimize"][0]["affinity"] for r in all_results]
    vina_dock = [r["vina"]["dock"][0]["affinity"] for r in all_results]

    n_eval = len(all_results)
    n_shards = len(pt_paths)
    n_samples_total = n_shards * 1000 if n_shards else None

    n_recon = n_complete = None
    for lp in log_paths:
        h, v = _parse_metrics_one_line(lp)
        if not h or "n_recon" not in h:
            continue
        ir, ic = h.index("n_recon"), h.index("n_complete")
        nr, nc = int(v[ir]), int(v[ic])
        n_recon = (n_recon or 0) + nr
        n_complete = (n_complete or 0) + nc
    if n_recon is None:
        n_recon = int(round(stab_accum["recon_success"] * (n_samples_total or 0)))
    if n_complete is None:
        n_complete = int(round(stab_accum["complete"] * (n_samples_total or 0)))

    jsd_all_mean = _mean_shard_metric(log_paths, "JSD_All_12A")
    jsd_cc_mean = _mean_shard_metric(log_paths, "JSD_CC_2A")

    out = {
        **stab_accum,
        **{k: float(c_bond_length_dict[k]) for k in sorted(c_bond_length_dict) if k.startswith("JSD_")},
        "JSD_All_12A": jsd_all_mean,
        "JSD_CC_2A": jsd_cc_mean,
        "atom_type_js": atom_type_js,
        "n_recon": n_recon,
        "n_complete": n_complete,
        "n_eval": n_eval,
        "QED_mean": float(np.mean(qed)) if qed else None,
        "QED_med": float(np.median(qed)) if qed else None,
        "SA_mean": float(np.mean(sa)) if sa else None,
        "SA_med": float(np.median(sa)) if sa else None,
        "similarity": Mean_pairwise_Tanimoto,
        "Diversity_mean": Diversity_mean,
        "Diversity_median": Diversity_median,
        "Vina_score_mean": float(np.mean(vina_score_only)),
        "Vina_score_median": float(np.median(vina_score_only)),
        "Vina_min_mean": float(np.mean(vina_min)),
        "Vina_min_median": float(np.median(vina_min)),
        "Dock_mean": float(np.mean(vina_dock)),
        "Dock_med": float(np.median(vina_dock)),
        "number": len(pocket_key_to_mols),
        "n_samples_total": n_samples_total,
    }
    return out


def _load_genbench_optional(path: str | None) -> dict[str, float | str]:
    """Map GenBench-style keys into our column names (partial)."""
    empty = {
        "Validity3D_pct": "",
        "Uniqueness3D_mean": "",
        "Diversity3D_mean": "",
        "Strain_energy_mean": "",
        "Strain_energy_median": "",
        "Clash_free_pct": "",
        "Centroid_dist_mean_A": "",
        "Diversity2D_pct": "",
        "Uniqueness2D_pct": "",
        "Validity2D_pct": "",
    }
    if not path or not os.path.exists(path):
        return empty
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pockets = data if isinstance(data, list) else [data]
    out_metrics: dict[str, list] = {}
    for res in pockets:
        for key, val in res.items():
            if key == "pocket":
                continue
            if key not in out_metrics:
                out_metrics[key] = []
            if isinstance(val, list):
                out_metrics[key].extend(v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v)))
            elif isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                out_metrics[key].append(val)

    def mean_pct(key_db: str, as_percent: bool) -> str:
        if key_db not in out_metrics or not out_metrics[key_db]:
            return ""
        m = statistics.mean(out_metrics[key_db])
        if as_percent:
            return "%.4f" % (m * 100.0) if m <= 1.0 else "%.4f" % m
        return "%.4f" % m

    def mean_plain(key_db: str) -> str:
        if key_db not in out_metrics or not out_metrics[key_db]:
            return ""
        return "%.4f" % statistics.mean(out_metrics[key_db])

    # Names follow get_genbench_report.py conventions
    upd = {
        "Validity3D_pct": mean_pct("Validity3D", True),
        "Uniqueness3D_mean": mean_plain("Uniqueness3D"),
        "Diversity3D_mean": mean_plain("Diversity3D"),
        "Strain_energy_mean": (
            "%.4f" % statistics.mean(out_metrics["Strain energy"]) if out_metrics.get("Strain energy") else ""
        ),
        "Strain_energy_median": (
            "%.4f" % statistics.median(out_metrics["Strain energy"]) if out_metrics.get("Strain energy") else ""
        ),
        "Clash_free_pct": (
            "%.4f"
            % (
                (sum(1 for x in out_metrics["Steric clash"] if x == 0) / len(out_metrics["Steric clash"])) * 100
            )
            if out_metrics.get("Steric clash")
            else ""
        ),
        "Centroid_dist_mean_A": (
            "%.4f" % statistics.mean(out_metrics["Distance to native centroid"])
            if out_metrics.get("Distance to native centroid")
            else ""
        ),
        "Diversity2D_pct": "",
        "Uniqueness2D_pct": "",
        "Validity2D_pct": "",
    }
    # Optional list-valued keys ending up as *_pct in GENBENCH export
    if "Diversity" in out_metrics and len(out_metrics["Diversity"]) == len(pockets):
        upd["Diversity2D_pct"] = "%.4f" % (statistics.mean(out_metrics["Diversity"]) * 100)
    if "Uniqueness" in out_metrics and len(out_metrics["Uniqueness"]) == len(pockets):
        upd["Uniqueness2D_pct"] = "%.4f" % (statistics.mean(out_metrics["Uniqueness"]) * 100)
    if "Validity" in out_metrics and len(out_metrics["Validity"]) == len(pockets):
        upd["Validity2D_pct"] = "%.4f" % (statistics.mean(out_metrics["Validity"]) * 100)

    empty.update(upd)
    return empty


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return "%.6f" % x
    return str(x)


def write_tsv(sample_dir: str, out_path: str, eval_step: int, genbench_json: str | None) -> None:
    m = merge_metrics(sample_dir, eval_step=eval_step)
    gb = _load_genbench_optional(genbench_json)

    headers = [
        "mol_stable",
        "atm_stable",
        "recon_success",
        "eval_success",
        "complete",
        "JSD_6-6|1",
        "JSD_6-6|2",
        "JSD_6-6|4",
        "JSD_6-7|1",
        "JSD_6-7|2",
        "JSD_6-7|4",
        "JSD_6-8|1",
        "JSD_6-8|2",
        "JSD_All_12A",
        "JSD_CC_2A",
        "atom_type_js",
        "n_recon",
        "n_complete",
        "n_eval",
        "QED_mean",
        "QED_med",
        "SA_mean",
        "SA_med",
        "similarity",
        "Diversity_mean",
        "Diversity_median",
        "Vina_score_mean",
        "Vina_score_median",
        "Vina_min_mean",
        "Vina_min_median",
        "Dock_mean",
        "Dock_med",
        "number",
        "Validity3D_pct",
        "Uniqueness3D_mean",
        "Diversity3D_mean",
        "Strain_energy_mean",
        "Strain_energy_median",
        "Clash_free_pct",
        "Centroid_dist_mean_A",
        "Diversity2D_pct",
        "Uniqueness2D_pct",
        "Validity2D_pct",
    ]

    row = [
        _fmt(m.get("mol_stable")),
        _fmt(m.get("atm_stable")),
        _fmt(m.get("recon_success")),
        _fmt(m.get("eval_success")),
        _fmt(m.get("complete")),
        _fmt(m.get("JSD_6-6|1")),
        _fmt(m.get("JSD_6-6|2")),
        _fmt(m.get("JSD_6-6|4")),
        _fmt(m.get("JSD_6-7|1")),
        _fmt(m.get("JSD_6-7|2")),
        _fmt(m.get("JSD_6-7|4")),
        _fmt(m.get("JSD_6-8|1")),
        _fmt(m.get("JSD_6-8|2")),
        _fmt(m.get("JSD_All_12A")),
        _fmt(m.get("JSD_CC_2A")),
        _fmt(m.get("atom_type_js")),
        str(m["n_recon"]),
        str(m["n_complete"]),
        str(m["n_eval"]),
        _fmt(m.get("QED_mean")),
        _fmt(m.get("QED_med")),
        _fmt(m.get("SA_mean")),
        _fmt(m.get("SA_med")),
        _fmt(m.get("similarity")),
        _fmt(m.get("Diversity_mean")),
        _fmt(m.get("Diversity_median")),
        _fmt(m.get("Vina_score_mean")),
        _fmt(m.get("Vina_score_median")),
        _fmt(m.get("Vina_min_mean")),
        _fmt(m.get("Vina_min_median")),
        _fmt(m.get("Dock_mean")),
        _fmt(m.get("Dock_med")),
        str(m.get("number", "")),
        gb["Validity3D_pct"],
        gb["Uniqueness3D_mean"],
        gb["Diversity3D_mean"],
        gb["Strain_energy_mean"],
        gb["Strain_energy_median"],
        gb["Clash_free_pct"],
        gb["Centroid_dist_mean_A"],
        gb["Diversity2D_pct"],
        gb["Uniqueness2D_pct"],
        gb["Validity2D_pct"],
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Merge eval_shards metrics into one TSV row.")
    ap.add_argument("sample_dir", type=str, help="quick_eval output dir containing eval_shards/")
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output TSV path (default: <sample_dir>/merged_metrics_<eval_step>.tsv)",
    )
    ap.add_argument("--eval_step", type=int, default=-1)
    ap.add_argument("--genbench-json", type=str, default=None, help="Optional aggregated GenBench JSON list file")
    args = ap.parse_args()
    sample_dir = os.path.abspath(args.sample_dir)
    out = args.output or os.path.join(sample_dir, "merged_metrics_%s.tsv" % args.eval_step)
    write_tsv(sample_dir, out, args.eval_step, args.genbench_json)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
