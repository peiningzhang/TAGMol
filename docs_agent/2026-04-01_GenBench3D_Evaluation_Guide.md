# GenBench3D Evaluation Guide for TAGMol

## Overview
This document outlines the workflow and fixes applied to run **GenBench3D** (a benchmark tool for structure-based molecular generation) on TAGMol generated results.

In this evaluation, we assess binding-related scores (when Vina is enabled inside GenBench3D) and molecular / 3D structural properties (Validity2D/3D, diversity, strain, clash, centroid distance, etc.) on the **CrossDocked**-style test set under `data/test_set`.

**Two Python environments**

- **TAGMol (`tagmol`)**: sampling, `evaluate_diffusion`, `quick_evaluate`, `get_genbench_report.py`, `run_genbench_eval.py` *driver* (the script itself can run here).
- **GenBench3D (`genbench3d` or your install)**: must be used to execute `sb_benchmark_mols.py` (needs **MDAnalysis**, RDKit, etc.). Do **not** rely on `python` from `tagmol` for that subprocess: use `--genbench_python` pointing at `.../envs/genbench3d/bin/python` (see §3). Installing full GenBench deps into `tagmol` (Python 3.8) is fragile for recent MDAnalysis dependency trees.

---

## 1. Preparing the SDF Inputs
TAGMol evaluation originally stores trajectories as `result_*.pt` under a sample directory. GenBench3D expects grouped **`*_generated.sdf`** (one multi-molecule SDF per pocket / target).

### Path A: Historical `export_to_sdf_grouped.py`
- **Extraction and grouping**: separate generated molecules into per-target SDFs.
- **Metadata**: optional properties (`QED`, `SA`, Vina from TAGMol) can be written into SDF fields.

```bash
conda activate tagmol
python export_to_sdf_grouped.py
```

Example layout: `experiments/trained_176000/sdfs_grouped/*_generated.sdf`.

### Path B: `quick_evaluate.py` + `evaluate_diffusion`
When you run quick eval with **`--export_grouped_sdf`**, TAGMol loads `eval_results/metrics_<eval_step>.pt` and writes the same grouped `*_generated.sdf` pattern (via `export_metrics_to_grouped_sdf` in ` scripts/quick_evaluate.py`). That flow requires RDKit and a successful prior evaluation pass that saved metrics.

---

## 2. GenBench3D Configuration Fixes
### Issue A: CrossDocked source file missing (`split_by_name.p`)
With `--source crossdocked`, GenBench3D may expect full CrossDocked training assets. **Practical default**: use **`-s ligboundconf`** (configured in `genbench3d/config/default.yaml` and passed by our runners) so reference geometry comes from the distributed LigBoundConf-based statistics—no full CrossDocked rebuild.

### Issue B: Native ligand / pocket file resolution
GenBench3D needs the native ligand SDF and receptor PDB per pocket. TAGMol uses names of the form `pocketdir_ligandbase`. **`run_genbench_eval.py`** and **`quick_evaluate.py`** (`run_genbench3d_suite`) both map `test_set/<pocket_dir>/*.sdf` and `*_rec.pdb` consistently.

### Issue C (patched GenBench3D): `--no_vina` and receptor preparation
If your `sb_benchmark_mols.py` uses **raw** `*_rec.pdb` when `--no_vina` is set, MDAnalysis → RDKit conversion for the pocket may fail (**no explicit hydrogens**). The robust pattern is to **still run `VinaProtein` / receptor preparation** to obtain a clean PDB for `Pocket`, while **skipping** `setup_vina` scoring when you want faster or duplicate-free runs. (Exact patch lives in your `genbench3d` tree, not in TAGMol.)

---

## 3. Running the Benchmark Pipeline

### 3.1 Standalone batch script: `run_genbench_eval.py`
Runs `sb_benchmark_mols.py` once per `*_generated.sdf`. **Important**: pass the GenBench-capable interpreter.

```bash
cd /shared/healthinfolab/phz24002/TAGMol

# Recommended (from tagmol env, but subprocess uses genbench3d python):
python run_genbench_eval.py \
  --do_conf_analysis \
  --no_vina \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python
```

**CLI (defaults match older hard-coded paths; override as needed)**

| Flag | Role |
|------|------|
| `--gen_dir` | Directory with `*_generated.sdf` (default: `experiments/trained_176000/sdfs_grouped`) |
| `--gb3d_dir` | GenBench3D repo root |
| `--test_set_dir` | Native ligand tree (default: `data/test_set`) |
| `--genbench_python` | Python for `sb_benchmark_mols.py` (default: env `GENBENCH_PYTHON` or `.../genbench3d/bin/python`) |
| `--do_conf_analysis` | Forwards to GenBench3D: Validity3D, strain, TFD-related metrics, etc. (slower) |
| `--no_vina` | Forwards to patched `sb_benchmark_mols.py`: skip Vina / minimized Vina (requires your GenBench patch) |

Aggregation writes **`all_results_aggregated.json`** using only **`results_*.json`**, so the aggregate file is not accidentally merged twice.

### 3.2 Integrated: `scripts/quick_evaluate.py`
After sampling + in-process `evaluate_diffusion`, you can run GenBench on the same run’s (or an existing) grouped SDF directory:

```bash
python scripts/quick_evaluate.py \
  --config <training_or_sampling.yml> \
  --checkpoint <checkpoint.pt> \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina
```

Useful flags:

- **`--sdf_grouped_dir`**: existing folder of `*_generated.sdf` (GenBench still runs; sampling always occurs in the current quick-eval design).
- **`--no_genbench_console_report`**: skip the boxed + TSV console summary from `get_genbench_report` (GenBench still runs).
- **`--test_set_dir`**: override native-ligand root for GenBench mapping.

**TAGMol vs GenBench Vina**: `evaluate_diffusion` docking uses `--docking_mode`; GenBench may still run internal Vina unless you pass **`--genbench_no_vina`** (and your GenBench supports it). Use `--docking_mode none` if you only want GenBench’s scores and no duplicate TAGMol-side Vina.

---

## 4. Console reporting and one-line summaries

### 4.1 `get_genbench_report.py`
Summarizes `all_results_aggregated.json` or a directory of `results_*.json`.

```bash
python get_genbench_report.py path/to/all_results_aggregated.json
```

**Aggregated “two-line” TSV** (after the human-readable box): the logger / stdout also emits:

- `GENBENCH_TWO_LINE_NAMES` + `GENBENCH_TWO_LINE_VALS` — tab-separated compact columns aligned with the boxed summary (e.g. `Validity3D_pct`, `Strain_energy_mean` / `Strain_energy_median`, clash-free, centroid, pocket-level 2D proportions as `*_pct`). If GenBench was run with `--no_vina`, Vina-related columns are omitted and a short note is printed.

**Strain energy aggregation**: `Strain_energy_mean` / `Strain_energy_median` are computed on the **pooled list** of all per-molecule strain values concatenated across pockets (from JSON lists), not “mean of per-pocket means” unless each pocket contributes a single scalar.

**`Uniqueness3D_mean` / `Diversity3D_mean`**: these are **our column labels** in the TSV for `statistics.mean` of the pooled `Uniqueness3D` / `Diversity3D` lists. They appear only when GenBench produced enough non-NaN values after aggregation; if every run was NaN, those columns are absent.

### 4.2 `quick_evaluate.py` end-of-run lines
Evaluation uses **in-process** `run_evaluation` + `report_evaluation_to_logger` from `scripts/evaluate_diffusion.py` (same text as CLI `evaluate_diffusion`). You still get:

- `METRICS_ONE_LINE_HEAD` / `METRICS_ONE_LINE_VAL` (tab-separated TAGMol metrics).

After GenBench (if enabled), the run **repeats** the TAGMol one-line pair once more for easy log scraping.

---

## 5. Scheduler sweep shell script
`scripts/run_quick_eval_schedulers_genbench3d.sh` loops `--time_scheduler` values and includes export + GenBench flags aligned with the project’s standard eval. Edit the embedded `training_*` log paths and conda Python path as needed.

---

## 6. Checking progress
Count finished pockets (one `results_*.json` per target):

```bash
ls /shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results/results_*.json 2>/dev/null | wc -l
```

Runtime is **much lower** when GenBench runs with **`--no_vina`**. With full Vina inside GenBench, budget on the order of tens of minutes for 100 pockets × many poses is reasonable on a typical GPU node.

---

## 7. Final aggregation artifact
`run_genbench_eval.py` writes:

`experiments/<run>/sdfs_grouped/genbench_results/all_results_aggregated.json`

(Exact path depends on `--gen_dir`.)

---

## 8. Example batch results (historical reference)
From an earlier full-Vina GenBench batch on `trained_176000` (numbers drift with model and flags):

| Metric | Result | Description |
| :--- | :--- | :--- |
| **High Affinity Rate** | **38.38%** | Share of poses with better relative minimized Vina than native |
| **Diversity2D** | **~88.8%** | Internal 2D diversity (see GenBench3D definitions) |
| **Clash-free Rate** | **78.13%** | Steric clash == 0 in GenBench’s metric |
| **Mean Minimized Vina** | **-5.487** | Pooled minimized scores (when Vina enabled) |
| **Centroid Distance** | **1.569 Å** | Mean distance to native centroid |

Regenerate any table with:

```bash
python get_genbench_report.py /path/to/all_results_aggregated.json
```

---

## 9. Troubleshooting quick reference

| Symptom | Likely cause | Action |
|--------|----------------|--------|
| `ModuleNotFoundError: MDAnalysis` in child process | Subprocess used `tagmol`’s `python` | Set `--genbench_python` to genbench3d env; see §3.1 |
| Pocket / RDKit error with `--no_vina` | Raw PDB without H for `Pocket` | Patch GenBench to use cleaned receptor PDB even when skipping Vina scoring |
| Empty `Uniqueness3D` / `Diversity3D` in TSV | All NaN in GenBench outputs | Normal for some settings; TSV omits those columns |
