import json
import os
import glob
import numpy as np

# Directory containing the JSON results
results_dir = "/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results"
json_files = glob.glob(os.path.join(results_dir, "results_*.json"))

if not json_files:
    print("No GenBench3D JSON result files found.")
    exit(1)

print(f"Found {len(json_files)} pocket results. Aggregating standard GenBench3D metrics...")

# We'll use this to keep track of any scalar metric found in the JSONs
all_metrics = {}

for f_path in json_files:
    with open(f_path, 'r') as f:
        try:
            res = json.load(f)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            continue
            
        for key, val in res.items():
            if key == 'pocket': continue
            
            if key not in all_metrics:
                all_metrics[key] = []
            
            if isinstance(val, list):
                # Filter out None and NaN
                valid_vals = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
                all_metrics[key].extend(valid_vals)
            elif isinstance(val, (int, float)):
                if not np.isnan(val):
                    all_metrics[key].append(val)

# Custom report output
print("\n" + "="*60)
print("      GenBench3D STANDARDIZED SUMMARY REPORT (Aggregated)")
print("="*60)

# 1. Connectivity / 3D Validity
if "Validity" in all_metrics:
    print(f"Molecular Graph Validity | {np.mean(all_metrics['Validity'])*100:.2f}%")

if "Validity3D" in all_metrics:
    print(f"3D Validity (Geometric)  | {np.mean(all_metrics['Validity3D'])*100:.2f}%")

# 2. Vina Performance
if "Minimized Vina score" in all_metrics:
    vals = all_metrics["Minimized Vina score"]
    print(f"Vina Score (Minimized)   | Mean: {np.mean(vals):.3f} | Median: {np.median(vals):.3f}")

if "Relative Min Vina score" in all_metrics:
    vals = np.array(all_metrics["Relative Min Vina score"])
    high_affinity_rate = (np.sum(vals < 0) / len(vals)) * 100
    print(f"High Affinity Rate       | {high_affinity_rate:.2f}% (Better than reference)")

# 3. Geometric metrics
if "Steric clash" in all_metrics:
    clash_vals = np.array(all_metrics["Steric clash"])
    clash_free_rate = (np.sum(clash_vals == 0) / len(clash_vals)) * 100
    print(f"Clash-free Rate          | {clash_free_rate:.2f}%")

if "Distance to native centroid" in all_metrics:
    print(f"Centroid Distance        | Mean: {np.mean(all_metrics['Distance to native centroid']):.3f} Å")

# 4. Other interesting percentages
print("-" * 60)
print("Structural Proportion / Patterns:")
for k in sorted(all_metrics.keys()):
    # Filter for things that looks like proportions (0-1) and aren't lists of per-mol values
    # usually these are summary metrics in GenBench JSON
    if len(all_metrics[k]) == len(json_files) and k not in ["Validity", "Validity3D"]:
        # Only print if it's a single value per pocket (summary stat)
        avg_val = np.mean(all_metrics[k])
        if 0 <= avg_val <= 1:
            print(f"  {k:22} | {avg_val*100:.2f}%")

print("="*60)
