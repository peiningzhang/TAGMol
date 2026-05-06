#!/bin/bash
#SBATCH --job-name=quick_eval
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/healthinfolab/phz24002/TAGMol/sbatch_script/logs/%x_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/TAGMol/sbatch_script/logs/%x_%j.err

# Quick evaluate on GPU (SLURM). Sweeps cfg_scale like scripts/run_quick_eval_schedulers_genbench3d.sh
#
# Usage:
#   cd /shared/healthinfolab/phz24002/TAGMol
#   sbatch sbatch_script/quick_evaluate_gpu.sh
#
# Override examples:
#   CFG_SCALES="1 2 3" DOCKING_MODE=vina_score sbatch sbatch_script/quick_evaluate_gpu.sh
#   OUTPUT_LOG_PREFIX=tmp_dock_log/my_sweep sbatch sbatch_script/quick_evaluate_gpu.sh
#
# Note: Do not background Python with & inside batch jobs; Slurm already runs one allocation.

set -euo pipefail

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

mkdir -p "$SCRIPT_DIR/sbatch_script/logs"

# Conda env for TAGMol (python + torch). GenBench uses a separate interpreter via --genbench_python.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/shared/healthinfolab/phz24002/anaconda3/envs/tagmol}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional: match cluster CUDA module if your PyTorch build expects it (uncomment if needed).
# module load cuda/11.7

# --- task parameters (override with env when calling sbatch) ---
CONFIG="${CONFIG:-logs_diffusion/training_cfg_muon_2026_04_17__13_40_04/sampling.yml}"
CHECKPOINT="${CHECKPOINT:-logs_diffusion/training_cfg_muon_2026_04_17__13_40_04/checkpoints/354000.pt}"
NUM_PROTEINS="${NUM_PROTEINS:-100}"
NUM_LIGANDS_PER_PROTEIN="${NUM_LIGANDS_PER_PROTEIN:-10}"
DOCKING_MODE="${DOCKING_MODE:-vina_dock}"
TIME_SCHEDULER="${TIME_SCHEDULER:-gen_arcsin}"

# Space-separated list; default matches scripts/run_quick_eval_schedulers_genbench3d.sh
if [[ -n "${CFG_SCALES:-}" ]]; then
  read -ra CFG_SCALE_VALUES <<< "$CFG_SCALES"
else
  CFG_SCALE_VALUES=(0 0.5 1.0 2.0 3.0 5.0 10.0 15.0 20.0)
fi

GB3D_DIR="${GB3D_DIR:-/shared/healthinfolab/phz24002/genbench3d}"
GENBENCH_PYTHON="${GENBENCH_PYTHON:-/home/phz24002/anaconda3/envs/genbench3d/bin/python}"

# Optional: set to e.g. ramp_up to mirror run_quick_eval_schedulers_genbench3d.sh
CFG_STRATEGY="${CFG_STRATEGY:-}"

# Per-cfg log prefix (relative to SCRIPT_DIR unless absolute). Each run writes ${prefix}_cfg${scale}.log
# OUTPUT_LOG_PREFIX overrides; else LOG_FILE (legacy) is used as prefix; else default stem includes scheduler name.
if [[ -n "${OUTPUT_LOG_PREFIX:-}" ]]; then
  LOG_PREFIX_REL="$OUTPUT_LOG_PREFIX"
elif [[ -n "${LOG_FILE:-}" ]]; then
  LOG_PREFIX_REL="$LOG_FILE"
else
  LOG_PREFIX_REL="tmp_dock_log/quick_eval_${TIME_SCHEDULER}"
fi

resolve_log_prefix() {
  local rel="$1"
  if [[ "$rel" = /* ]]; then
    echo "$rel"
  else
    echo "$SCRIPT_DIR/$rel"
  fi
}

LOG_PREFIX_BASE="$(resolve_log_prefix "$LOG_PREFIX_REL")"
mkdir -p "$(dirname "$LOG_PREFIX_BASE")"

echo "=========================================="
echo "quick_evaluate (SLURM) - cfg_scale sweep"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "WORKDIR: $SCRIPT_DIR"
echo "TIME_SCHEDULER: $TIME_SCHEDULER"
echo "CFG scales: ${CFG_SCALE_VALUES[*]}"
echo "Log prefix: $LOG_PREFIX_BASE _cfg<scale>.log"
echo "=========================================="

for CFG in "${CFG_SCALE_VALUES[@]}"; do
  echo ""
  echo "=========================================================="
  echo "Running quick_evaluate: time_scheduler=$TIME_SCHEDULER cfg_scale=$CFG"
  echo "=========================================================="
  LOG_ABS="${LOG_PREFIX_BASE}_cfg${CFG}.log"

  CFG_CMD=(python scripts/quick_evaluate.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --num_proteins "$NUM_PROTEINS"
    --num_ligands_per_protein "$NUM_LIGANDS_PER_PROTEIN"
    --docking_mode "$DOCKING_MODE"
    --export_grouped_sdf
    --run_genbench
    --gb3d_dir "$GB3D_DIR"
    --genbench_python "$GENBENCH_PYTHON"
    --genbench_do_conf_analysis
    --genbench_no_vina
    --time_scheduler "$TIME_SCHEDULER"
    --cfg_scale "$CFG"
  )
  if [[ -n "$CFG_STRATEGY" ]]; then
    CFG_CMD+=(--cfg_strategy "$CFG_STRATEGY")
  fi

  "${CFG_CMD[@]}" > "$LOG_ABS" 2>&1

  echo "Done: time_scheduler=$TIME_SCHEDULER cfg_scale=$CFG - wrote $LOG_ABS"
done

echo ""
echo "Finished all cfg_scale runs: $(date)"
