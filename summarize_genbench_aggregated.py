#!/usr/bin/env python3
"""
Summarize GenBench3D aggregated JSON: hit rates, docking stats, optional intra-pocket
diversity from grouped SDFs (Morgan fingerprints, mean pairwise Tanimoto).
"""
import argparse
import json
import math
import os
import sys
from statistics import mean, median
from typing import Any, Dict, List, Optional


def rdkit_available():
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def pocket_stem_from_record(pocket_field: str) -> str:
    base = os.path.basename(pocket_field)
    if base.startswith("results_") and base.endswith(".json"):
        return base[len("results_") : -len(".json")]
    if base.endswith(".json"):
        return base[: -len(".json")]
    return base


def hit_fraction(scores: List[float], threshold: float, lower_is_better: bool = True) -> float:
    """Vina: lower (more negative) is better -> hit if score <= threshold."""
    if not scores:
        return float("nan")
    if lower_is_better:
        return sum(1 for s in scores if s <= threshold) / len(scores)
    return sum(1 for s in scores if s >= threshold) / len(scores)


def best_per_pocket(scores: List[float]) -> Optional[float]:
    if not scores:
        return None
    return min(scores)


def mean_pairwise_tanimoto(fps) -> float:
    from rdkit import DataStructs

    n = len(fps)
    if n < 2:
        return float("nan")
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += DataStructs.TanimotoSimilarity(fps[i], fps[j])
            cnt += 1
    return total / cnt


def load_fingerprints_sdf(path: str, n_expected: Optional[int] = None):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    suppl = Chem.SDMolSupplier(path, removeHs=False)
    mols = [m for m in suppl if m is not None]
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
    if n_expected is not None and len(fps) != n_expected:
        return None, f"SDF count {len(fps)} != JSON list length {n_expected}"
    return fps, None


def summarize_record(
    rec: Dict[str, Any],
    sdf_dir: Optional[str],
    hit_thresholds: List[float],
) -> Dict[str, Any]:
    pocket_key = rec.get("pocket", "")
    stem = pocket_stem_from_record(str(pocket_key))

    vina = rec.get("Vina score") or []
    min_vina = rec.get("Minimized Vina score") or []
    rel = rec.get("Relative Vina score") or []
    rel_min = rec.get("Relative Min Vina score") or []
    clash = rec.get("Steric clash") or []
    dist = rec.get("Distance to native centroid") or []

    n = len(min_vina) if min_vina else len(vina)

    out: Dict[str, Any] = {
        "pocket": pocket_key,
        "stem": stem,
        "n_molecules": n,
        "best_minimized_vina": best_per_pocket(min_vina) if min_vina else None,
        "best_vina": best_per_pocket(vina) if vina else None,
        "mean_minimized_vina": mean(min_vina) if min_vina else None,
        "mean_vina": mean(vina) if vina else None,
        "median_minimized_vina": median(min_vina) if min_vina else None,
        "hit_rate_minimized": {},
        "hit_rate_vina": {},
        "mean_steric_clash": mean(clash) if clash else None,
        "frac_zero_clash": sum(1 for c in clash if c == 0) / len(clash) if clash else None,
        "mean_dist_native_A": mean(dist) if dist else None,
        "median_dist_native_A": median(dist) if dist else None,
        "mean_pairwise_tanimoto_morgan2": None,
        "diversity_1_minus_mean_tanimoto": None,
        "diversity_note": None,
    }

    for t in hit_thresholds:
        out["hit_rate_minimized"][str(t)] = hit_fraction(min_vina, t) if min_vina else float("nan")
        out["hit_rate_vina"][str(t)] = hit_fraction(vina, t) if vina else float("nan")

    # Relative minimized: negative or zero often means better than / on par with native (context-dependent)
    if rel_min:
        out["mean_relative_min_vina"] = mean(rel_min)
        out["frac_relative_min_le_0"] = sum(1 for x in rel_min if x <= 0) / len(rel_min)

    if sdf_dir and stem:
        sdf_path = os.path.join(sdf_dir, f"{stem}_generated.sdf")
        if os.path.isfile(sdf_path):
            try:
                fps, err = load_fingerprints_sdf(sdf_path, n_expected=n if n else None)
                if err:
                    out["diversity_note"] = err
                elif fps and len(fps) >= 2:
                    mp = mean_pairwise_tanimoto(fps)
                    out["mean_pairwise_tanimoto_morgan2"] = mp
                    out["diversity_1_minus_mean_tanimoto"] = 1.0 - mp
                elif fps and len(fps) == 1:
                    out["diversity_note"] = "single molecule; no pairwise similarity"
            except Exception as e:
                out["diversity_note"] = str(e)
        else:
            out["diversity_note"] = f"missing SDF: {sdf_path}"

    return out


def nanmean(vals: List[float]) -> float:
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return mean(xs) if xs else float("nan")


def print_report(per_pocket: List[Dict[str, Any]], hit_thresholds: List[float]) -> None:
    print("--- Hit rate (pocket-level: best minimized Vina in pocket <= threshold) ---")
    bests = [p["best_minimized_vina"] for p in per_pocket if p["best_minimized_vina"] is not None]
    for t in hit_thresholds:
        pocket_hits = sum(1 for b in bests if b <= t) / len(bests) if bests else float("nan")
        print(f"  threshold {t:5.1f}: {pocket_hits * 100:5.1f}% of pockets ({sum(1 for b in bests if b <= t)}/{len(bests)})")
    print()

    print("--- Per-pocket averages (macro mean over pockets) ---")
    macro_mean_min = nanmean([p["mean_minimized_vina"] for p in per_pocket if p["mean_minimized_vina"] is not None])
    macro_mean_vina = nanmean([p["mean_vina"] for p in per_pocket if p["mean_vina"] is not None])
    macro_mean_dist = nanmean([p["mean_dist_native_A"] for p in per_pocket if p["mean_dist_native_A"] is not None])
    macro_frac_clash0 = nanmean([p["frac_zero_clash"] for p in per_pocket if p["frac_zero_clash"] is not None])
    print(f"  Mean of per-pocket mean Minimized Vina: {macro_mean_min: .3f}")
    print(f"  Mean of per-pocket mean Vina:           {macro_mean_vina: .3f}")
    print(f"  Mean of per-pocket mean centroid dist (Å): {macro_mean_dist: .3f}")
    print(f"  Mean of per-pocket fraction zero steric clash: {macro_frac_clash0 * 100: .1f}%")
    div_vals = [
        p["diversity_1_minus_mean_tanimoto"]
        for p in per_pocket
        if p["diversity_1_minus_mean_tanimoto"] is not None
        and not (isinstance(p["diversity_1_minus_mean_tanimoto"], float) and math.isnan(p["diversity_1_minus_mean_tanimoto"]))
    ]
    if div_vals:
        print(f"  Mean intra-pocket diversity (1 - mean Tanimoto Morgan r=2): {mean(div_vals): .4f}")
        print(f"  (higher => more diverse; computed from grouped SDFs when available)")
    print()

    print("--- Worst / best pockets by best Minimized Vina ---")
    ranked = sorted(
        [p for p in per_pocket if p["best_minimized_vina"] is not None],
        key=lambda x: x["best_minimized_vina"],
    )
    print("  Best 5 pockets (lowest best Minimized Vina):")
    for p in ranked[:5]:
        print(f"    {p['best_minimized_vina']:7.3f}  {p['stem'][:64]}")
    print("  Worst 5 pockets:")
    for p in ranked[-5:]:
        print(f"    {p['best_minimized_vina']:7.3f}  {p['stem'][:64]}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize all_results_aggregated.json from GenBench3D.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results/all_results_aggregated.json",
        help="Path to all_results_aggregated.json",
    )
    parser.add_argument(
        "--sdf-dir",
        default=None,
        help="Directory with *_generated.sdf files (default: parent of genbench_results)",
    )
    parser.add_argument(
        "--thresholds",
        default="-10,-9,-8,-7,-6",
        help="Comma-separated Vina thresholds (hits if score <= threshold)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write full numeric summary JSON",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Skip RDKit / SDF fingerprint diversity",
    )
    args = parser.parse_args()

    hit_thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    with open(args.json_path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    genbench_dir = os.path.dirname(os.path.abspath(args.json_path))
    sdf_dir = args.sdf_dir or os.path.dirname(genbench_dir)
    do_fp = (not args.no_diversity) and rdkit_available()
    if not args.no_diversity and not rdkit_available():
        print(
            "\nNote: RDKit is not importable in this Python; intra-pocket diversity was skipped. "
            "Run the same script in your conda env (e.g. the one used for TAGMol) to get Morgan/Tanimoto stats.\n"
        )
    sdf_dir_for_fp = sdf_dir if do_fp else None

    per_pocket = [summarize_record(rec, sdf_dir_for_fp, hit_thresholds) for rec in data]

    # Pooled molecule-level metrics
    all_min_vina: List[float] = []
    all_vina: List[float] = []
    for rec in data:
        for x in rec.get("Minimized Vina score") or []:
            all_min_vina.append(float(x))
        for x in rec.get("Vina score") or []:
            all_vina.append(float(x))

    print("=" * 72)
    print("GenBench3D aggregated summary")
    print("=" * 72)
    print(f"Input: {args.json_path}")
    print(f"Pockets: {len(data)}  |  Total molecules (poses): {len(all_min_vina)}")
    print()
    print("--- Hit rate (molecule-level, all pockets pooled) ---")
    for t in hit_thresholds:
        hr_min = hit_fraction(all_min_vina, t)
        hr_v = hit_fraction(all_vina, t)
        print(f"  Minimized Vina <= {t:5.1f}: {hr_min * 100:5.1f}%  ({sum(1 for s in all_min_vina if s <= t)}/{len(all_min_vina)})")
        print(f"  Raw Vina       <= {t:5.1f}: {hr_v * 100:5.1f}%  ({sum(1 for s in all_vina if s <= t)}/{len(all_vina)})")
    print()

    print_report(per_pocket, hit_thresholds)

    if args.out_json:
        def _json_safe(o: Any) -> Any:
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_json_safe(v) for v in o]
            return o

        out = _json_safe(
            {
                "source_json": os.path.abspath(args.json_path),
                "sdf_dir": sdf_dir,
                "diversity_computed": do_fp,
                "hit_thresholds": hit_thresholds,
                "n_pockets": len(data),
                "n_molecules": len(all_min_vina),
                "pooled_hit_rate_minimized_vina": {str(t): hit_fraction(all_min_vina, t) for t in hit_thresholds},
                "pooled_hit_rate_vina": {str(t): hit_fraction(all_vina, t) for t in hit_thresholds},
                "per_pocket": per_pocket,
            }
        )
        with open(args.out_json, "w") as wf:
            json.dump(out, wf, indent=2)
        print(f"Wrote JSON summary: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
