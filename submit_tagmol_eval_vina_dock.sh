#!/bin/bash
# Submit 10 CPU jobs: each runs evaluate_diffusion (vina_dock) on 10 pockets (100 total).
#
# Usage:
#   bash submit_tagmol_eval_vina_dock.sh [SAMPLE_DIR]
# Default SAMPLE_DIR (override as first arg):
#   /shared/healthinfolab/phz24002/TAGMol/tmp_eval/quick_eval_2026_04_22__13_12_18_p9lplts7
#
# Optional (export before running):
#   PROTEIN_ROOT  — same as quick_evaluate / evaluate_diffusion (default: REPO/data/test_set)
#   EVAL_STEP     — default -1
#   EXHAUSTIVENESS — default 16
#   EVAL_PY       — python path (else conda tagmol from tagmol_eval_vina_dock_slurm.sh)
#
# After completion you get 10 files:
#   $SAMPLE_DIR/eval_shards/shard_0_9/eval_results/metrics_*.pt
#   ...
#   $SAMPLE_DIR/eval_shards/shard_90_99/eval_results/metrics_*.pt
# Each file's all_results covers only that shard; bond_length / JSD in each file are for that
# subset only. To match a single full quick_eval run, concatenate all_results and re-aggregate
# (or load all 10 in Python and merge).

set -euo pipefail

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

SAMPLE_DIR="${1:-/shared/healthinfolab/phz24002/TAGMol/tmp_eval/quick_eval_2026_04_22__13_12_18_p9lplts7}"
SAMPLE_DIR=$(readlink -f "$SAMPLE_DIR")

mkdir -p "$SCRIPT_DIR/logs"

if [ ! -d "$SAMPLE_DIR" ]; then
  echo "ERROR: SAMPLE_DIR not found: $SAMPLE_DIR" >&2
  exit 1
fi

# 10 jobs × 10 pockets: 0–9, 10–19, …, 90–99
for i in $(seq 0 9); do
  START=$((i * 10))
  END=$((START + 9))
  echo "Submitting eval job shard $START-$END..."
  sbatch --export=ALL --job-name=eval_dock_${START}-${END} \
    tagmol_eval_vina_dock_slurm.sh "$SAMPLE_DIR" "$START" "$END"
done

echo ""
echo "Submitted 10 jobs (each \"Submitted batch job <JOBID>\" line from sbatch means success)."
echo "Check queue:  squeue -u \"$USER\""
echo "If squeue is empty: jobs may have finished (or failed fast). See:  sacct -S today -u \"$USER\" | head -30"
echo "Per-job logs:   $SCRIPT_DIR/logs/eval_dock_<JOBID>.out  and  .err"
echo "Results:        $SAMPLE_DIR/eval_shards/shard_*/eval_results/"
