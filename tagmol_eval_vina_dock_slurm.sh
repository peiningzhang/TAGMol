#!/bin/bash
#SBATCH --job-name=tagmol_eval_dock
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# 10 pockets × 100 ligands, Vina + fork may spike; raise if sacct shows FAILED in ~0–5s (OOM/53).
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/healthinfolab/phz24002/TAGMol/logs/eval_dock_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/TAGMol/logs/eval_dock_%j.err

# Evaluate TAGMol quick_eval / sample output with Vina (CPU). One job = one index range of result_*.pt.
#
# Usage (direct):
#   sbatch tagmol_eval_vina_dock_slurm.sh <SAMPLE_DIR> <start_idx> <end_idx>
# Example:
#   sbatch tagmol_eval_vina_dock_slurm.sh /path/to/tmp_eval/quick_eval_xxx 0 9
#
# Optional env (before sbatch or use sbatch --export=ALL,VAR=val):
#   PROTEIN_ROOT   Receptor / pocket tree root (default: $REPO/data/test_set)
#   EVAL_STEP      --eval_step (default: -1)
#   EXHAUSTIVENESS (default: 16)
#   EVAL_PY        python binary (default: conda tagmol)

set -euo pipefail

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

SAMPLE_DIR="${1:-}"
START_IDX="${2:-}"
END_IDX="${3:-}"

if [ -z "$SAMPLE_DIR" ] || [ -z "$START_IDX" ] || [ -z "$END_IDX" ]; then
  echo "Usage: $0 <SAMPLE_DIR> <start_idx> <end_idx>" >&2
  exit 1
fi

SAMPLE_DIR=$(readlink -f "$SAMPLE_DIR")
PROTEIN_ROOT="${PROTEIN_ROOT:-$SCRIPT_DIR/data/test_set}"
PROTEIN_ROOT=$(readlink -f "$PROTEIN_ROOT")
EVAL_STEP="${EVAL_STEP:--1}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-16}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/shared/healthinfolab/phz24002/anaconda3/envs/tagmol}"
EVAL_PY="${EVAL_PY:-$CONDA_ENV_PATH/bin/python}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
# set -u: append to PYTHONPATH only if set (Slurm jobs often have no PYTHONPATH)
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SHARD_DIR="$SAMPLE_DIR/eval_shards/shard_${START_IDX}_${END_IDX}"
mkdir -p "$SHARD_DIR" "$SCRIPT_DIR/logs"

echo "=========================================="
echo "TAGMol evaluate_diffusion (vina_dock)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Sample dir: $SAMPLE_DIR"
echo "Shard: $SHARD_DIR"
echo "Pocket indices: $START_IDX - $END_IDX"
echo "protein_root: $PROTEIN_ROOT"
echo "eval_step: $EVAL_STEP  exhaustiveness: $EXHAUSTIVENESS"
echo "Started: $(date)"
echo "=========================================="

for i in $(seq "$START_IDX" "$END_IDX"); do
  src="$SAMPLE_DIR/result_${i}.pt"
  if [ ! -f "$src" ]; then
    echo "ERROR: missing $src" >&2
    exit 1
  fi
  ln -sf "$src" "$SHARD_DIR/result_${i}.pt"
done

"$EVAL_PY" scripts/evaluate_diffusion.py "$SHARD_DIR" \
  --docking_mode vina_dock \
  --protein_root "$PROTEIN_ROOT" \
  --eval_step "$EVAL_STEP" \
  --exhaustiveness "$EXHAUSTIVENESS" \
  --eval_num_examples $((END_IDX - START_IDX + 1)) \
  --docking_isolate_process True \
  --verbose True \
  --save True \
  --one_line

echo ""
echo "Finished: $(date)"
echo "Per-shard metrics: $SHARD_DIR/eval_results/metrics_${EVAL_STEP}.pt"
echo "Note: global metrics over all 100 pockets require merging shard metrics (see docstring in submit script)."
