# GenBench3D Evaluation Guide for TAGMol

## Overview
This document outlines the workflow and fixes applied to run **GenBench3D** (a benchmark tool for structure-based molecular generation) on TAGMol generated results. 

In this evaluation, we are assessing hitting rates (like Vina affinity scores) and molecular structural properties (Diversity, SA, QED, etc.) across the **CrossDocked** 100-target test set.

## 1. Preparing the SDF Inputs
TAGMol originally outputs evaluation results as PyTorch Tensor `.pt` files (e.g. `metrics_-1.pt`). GenBench3D requires valid 3D `.sdf` files organized appropriately.

- **Extraction and Grouping**: Use `export_to_sdf_grouped.py`. It separates the generated molecules into 100 targets.
- **Metadata**: Key properties (`QED`, `SA`, `Vina`) are serialized into the SDF.

**Execution Command**:
```bash
conda activate tagmol
python export_to_sdf_grouped.py
```

- **Output Location**: `/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/`

## 2. GenBench3D Configuration Fixes
Several issues natively tied to GenBench3D's source lookup were encountered and successfully fixed:

### Issue A: CrossDocked Source File Missing (`split_by_name.p`)
When specifying `--source crossdocked`, GenBench3D attempts to rebuild and load Reference Geometry (bond lengths, angles) from the whole `CrossDocked` dataset using `split_by_name.p`.
* **Fix implemented**: We updated the `genbench3d/config/default.yaml` logic to bypass attempting to parse missing `crossdocked` full-training `.sdf` bundles, switching the arguments to natively point to `-s ligboundconf`. This leverages pre-computed, high-quality metric distributions distributed by GenBench3D, avoiding `FileNotFoundError` crashes natively.

### Issue B: TAGMol Native Ligand Naming Prefix Mismatch
GenBench3D requires the explicit original structural coordinates of the Native Ligand. TAGMol's dictionary previously concatenated the pocket dir name + ligand base name, which prevented glob searches for the correct ligand in `TAGMol/data/test_set`.
* **Fix implemented**: A robust mapping logic was integrated into the runner script `run_genbench_eval.py`. It accurately splits and searches the subdirectories of `TAGMol/data/test_set/` to extract absolute paths for both `*_rec.pdb` and `*.sdf` flawlessly.

## 3. Running the Benchmark Pipeline
To evaluate out the 100 grouped targets:

```bash
ssh gpu35
conda_home
conda activate genbench3d

cd /shared/healthinfolab/phz24002/TAGMol
python run_genbench_eval.py
```

### Checking the Progress
The evaluation is Vina-computation-heavy (averaging 30-40 mins for all 1000 compounds). 
You can monitor the exact status of target evaluations completed at any time by listing the outputs:
```bash
ls /shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results/ | wc -l
```
*(Every generated `results_*.json` file represents 1 fully completed target/pocket.)*

## 4. Final Aggregation
Upon 100% completion, `run_genbench_eval.py` automatically concatenates all 100 mini JSON outputs into a centralized summary:
👈 `/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results/all_results_aggregated.json`

## 5. Typical Results Summary
Based on the batch `trained_176000`, the following metrics were extracted using `get_genbench_report.py`:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **High Affinity Rate** | **38.38%** | Percentage of molecules binding better than reference |
| **Diversity** | **0.8883** | Internal Tanimoto diversity (1 - avg similarity) |
| **Clash-free Rate** | **78.13%** | 3D validity based on steric hindrance |
| **Mean Vina Score** | **-5.487** | Average binding affinity (minimized) |
| **Centroid Distance** | **1.569 Å** | Average deviation from native ligand position |

To regenerate this summary table, run:
```bash
python get_genbench_report.py
```

