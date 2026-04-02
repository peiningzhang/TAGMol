#!/usr/bin/env python3
"""
Print a compact GenBench3D summary from either:
  - all_results_aggregated.json (list of per-pocket dicts), or
  - a genbench_results directory of results_*.json

For hit rates / diversity from the same JSON, see summarize_genbench_aggregated.py.
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def load_pocket_dicts(path_or_dir):
    """Return list of per-pocket result dicts."""
    path_or_dir = os.path.abspath(path_or_dir)
    if os.path.isfile(path_or_dir):
        with open(path_or_dir, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("JSON root must be a list or dict, got %s" % type(data))
    if os.path.isdir(path_or_dir):
        json_files = sorted(glob.glob(os.path.join(path_or_dir, "results_*.json")))
        if not json_files:
            raise FileNotFoundError("No results_*.json under %s" % path_or_dir)
        out = []
        for f_path in json_files:
            with open(f_path, "r") as f:
                out.append(json.load(f))
        return out
    raise FileNotFoundError("Not a file or directory: %s" % path_or_dir)


def aggregate_metrics(pockets):
    all_metrics = {}
    n_pockets = len(pockets)
    for res in pockets:
        for key, val in res.items():
            if key == "pocket":
                continue
            if key not in all_metrics:
                all_metrics[key] = []
            if isinstance(val, list):
                valid_vals = [v for v in val if v is not None and not _is_nan(v)]
                all_metrics[key].extend(valid_vals)
            elif isinstance(val, (int, float)):
                if not _is_nan(val):
                    all_metrics[key].append(val)
    return all_metrics, n_pockets


def genbench_summary_two_line_tsv(all_metrics, n_pockets):
    """Two tab-separated lines (headers, values) for the same scalars as print_report (aggregated)."""
    cols = []
    cols.append(("n_pockets", str(n_pockets)))
    if "Validity" in all_metrics and all_metrics["Validity"]:
        cols.append(("Validity_graph_pct", "%.2f" % (statistics.mean(all_metrics["Validity"]) * 100)))
    if "Validity3D" in all_metrics and all_metrics["Validity3D"]:
        cols.append(("Validity3D_pct", "%.2f" % (statistics.mean(all_metrics["Validity3D"]) * 100)))
    for label, key in (
        ("Uniqueness3D_mean", "Uniqueness3D"),
        ("Diversity3D_mean", "Diversity3D"),
        ("Novelty3D_mean", "Novelty3D"),
    ):
        if key in all_metrics and all_metrics[key]:
            cols.append((label, "%.4f" % statistics.mean(all_metrics[key])))
    if "Strain energy" in all_metrics and all_metrics["Strain energy"]:
        se = all_metrics["Strain energy"]
        cols.append(("Strain_energy_mean", "%.3f" % statistics.mean(se)))
        cols.append(("Strain_energy_median", "%.3f" % statistics.median(se)))
    if "Minimized Vina score" in all_metrics and all_metrics["Minimized Vina score"]:
        vals = all_metrics["Minimized Vina score"]
        cols.append(("Vina_min_mean", "%.3f" % statistics.mean(vals)))
        cols.append(("Vina_min_median", "%.3f" % statistics.median(vals)))
    if "Relative Min Vina score" in all_metrics and all_metrics["Relative Min Vina score"]:
        vals = all_metrics["Relative Min Vina score"]
        hr = (sum(1 for x in vals if x < 0) / len(vals)) * 100 if vals else 0.0
        cols.append(("High_affinity_rate_pct", "%.2f" % hr))
    if "Steric clash" in all_metrics and all_metrics["Steric clash"]:
        clash_vals = all_metrics["Steric clash"]
        rate = (sum(1 for x in clash_vals if x == 0) / len(clash_vals)) * 100 if clash_vals else 0.0
        cols.append(("Clash_free_pct", "%.2f" % rate))
    if "Distance to native centroid" in all_metrics and all_metrics["Distance to native centroid"]:
        cols.append(
            ("Centroid_dist_mean_A", "%.3f" % statistics.mean(all_metrics["Distance to native centroid"]))
        )
    json_files_count = n_pockets
    for k in sorted(all_metrics.keys()):
        if len(all_metrics[k]) == json_files_count and k not in ("Validity", "Validity3D"):
            avg_val = statistics.mean(all_metrics[k])
            if 0 <= avg_val <= 1:
                cols.append(("%s_pct" % k, "%.2f" % (avg_val * 100)))
    if not cols:
        return "", ""
    names = "\t".join(c[0] for c in cols)
    vals = "\t".join(c[1] for c in cols)
    return names, vals


def log_genbench_two_line_metrics(logger, all_metrics, n_pockets):
    """Emit GENBENCH_TWO_LINE_NAMES / GENBENCH_TWO_LINE_VALS for aggregated metrics."""
    g_names, g_vals = genbench_summary_two_line_tsv(all_metrics, n_pockets)
    if g_names:
        logger.info("GENBENCH_TWO_LINE_NAMES\t%s", g_names)
        logger.info("GENBENCH_TWO_LINE_VALS\t%s", g_vals)


def print_report(all_metrics, n_pockets):
    print("\n" + "=" * 60)
    print("      GenBench3D STANDARDIZED SUMMARY REPORT (Aggregated)")
    print("=" * 60)
    print("Pockets in summary: %d" % n_pockets)

    if "Validity" in all_metrics:
        v = all_metrics["Validity"]
        print("Molecular Graph Validity | %.2f%%" % (statistics.mean(v) * 100))

    if "Validity3D" in all_metrics:
        v = all_metrics["Validity3D"]
        print("3D Validity (Geometric)  | %.2f%%" % (statistics.mean(v) * 100))

    for _label, _key in (
        ("Uniqueness3D (TFD)", "Uniqueness3D"),
        ("Diversity3D (TFD)", "Diversity3D"),
        ("Novelty3D (TFD)", "Novelty3D"),
    ):
        if _key in all_metrics and all_metrics[_key]:
            vals = all_metrics[_key]
            print("%-26s | Mean: %.4f" % (_label, statistics.mean(vals)))

    if "Strain energy" in all_metrics and all_metrics["Strain energy"]:
        se = all_metrics["Strain energy"]
        print(
            "Strain energy (MMFF94s)   | Mean: %.3f | Median: %.3f"
            % (statistics.mean(se), statistics.median(se))
        )

    if "Minimized Vina score" in all_metrics:
        vals = all_metrics["Minimized Vina score"]
        print(
            "Vina Score (Minimized)   | Mean: %.3f | Median: %.3f"
            % (statistics.mean(vals), statistics.median(vals))
        )

    if "Relative Min Vina score" in all_metrics:
        vals = all_metrics["Relative Min Vina score"]
        high_affinity_rate = (sum(1 for x in vals if x < 0) / len(vals)) * 100 if vals else 0.0
        print("High Affinity Rate       | %.2f%% (Relative Min Vina < 0 vs native)" % high_affinity_rate)

    if "Steric clash" in all_metrics:
        clash_vals = all_metrics["Steric clash"]
        clash_free_rate = (sum(1 for x in clash_vals if x == 0) / len(clash_vals)) * 100 if clash_vals else 0.0
        print("Clash-free Rate          | %.2f%%" % clash_free_rate)

    if "Distance to native centroid" in all_metrics:
        d = all_metrics["Distance to native centroid"]
        print("Centroid Distance        | Mean: %.3f Å" % statistics.mean(d))

    print("-" * 60)
    print("Structural Proportion / Patterns (single scalar per pocket, 0–1):")
    json_files_count = n_pockets
    for k in sorted(all_metrics.keys()):
        if len(all_metrics[k]) == json_files_count and k not in ("Validity", "Validity3D"):
            avg_val = statistics.mean(all_metrics[k])
            if 0 <= avg_val <= 1:
                print("  %-22s | %.2f%%" % (k, avg_val * 100))

    print("=" * 60)
    if "Minimized Vina score" in all_metrics or "Vina score" in all_metrics:
        print(
            "Note: GenBench3D Vina is computed inside the benchmark (not TAGMol evaluate_diffusion).\n"
            "      To skip duplicate Vina in quick_eval, use --docking_mode none when you only need GenBench scores."
        )
    else:
        print(
            "Note: No Vina fields in these results (e.g. GenBench was run with --no_vina / quick_eval --genbench_no_vina)."
        )
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize GenBench3D results from all_results_aggregated.json or a results directory."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to all_results_aggregated.json, or a genbench_results directory with results_*.json",
    )
    parser.add_argument(
        "--aggregated",
        type=str,
        default=None,
        help="Explicit path to all_results_aggregated.json (overrides positional)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        dest="results_dir",
        help="Directory containing only results_*.json (use if you did not write aggregated JSON)",
    )
    args = parser.parse_args()

    target = args.aggregated or args.results_dir or args.input_path
    if not target:
        parser.error(
            "Provide a path: e.g. python get_genbench_report.py path/to/all_results_aggregated.json\n"
            "   or: python get_genbench_report.py --dir path/to/genbench_results"
        )

    try:
        pockets = load_pocket_dicts(target)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print("Error: %s" % e, file=sys.stderr)
        return 1

    if not pockets:
        print("No pocket results loaded.")
        return 1

    print("Loaded %d pocket result(s) from %s" % (len(pockets), os.path.abspath(target)))
    all_metrics, n_pockets = aggregate_metrics(pockets)
    print_report(all_metrics, n_pockets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
