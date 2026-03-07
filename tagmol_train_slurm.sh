#!/bin/bash
#SBATCH --job-name=tagmol_train
#SBATCH --partition=priority-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/shared/healthinfolab/phz24002/TAGMol/logs/train_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/TAGMol/logs/train_%j.err
#SBATCH --constraint="a100"

# TAGMol Training Script for SLURM
# Usage: sbatch tagmol_train_slurm.sh [diffusion|guide_ba|guide_qed|guide_sa]

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

JOB_TYPE=${1:-diffusion}
RETRY_COUNT=${2:-0}
MAX_RETRIES=3

CONDA_ENV_PATH="/shared/healthinfolab/phz24002/anaconda3/envs/tagmol"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

mkdir -p "$SCRIPT_DIR/logs"

echo "=========================================="
echo "TAGMol Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Type: $JOB_TYPE"
echo "Retry attempt: $RETRY_COUNT / $MAX_RETRIES"
echo "Started: $(date)"
echo "=========================================="

case $JOB_TYPE in
  diffusion)
    CONFIG="$SCRIPT_DIR/configs/training.yml"
    SCRIPT="$SCRIPT_DIR/scripts/train_diffusion.py"
    echo "Training Diffusion Model"
    echo "Config: $CONFIG"
    python "$SCRIPT" "$CONFIG"
    ;;
  
  guide_ba)
    CONFIG="$SCRIPT_DIR/configs/training_dock_guide.yml"
    SCRIPT="$SCRIPT_DIR/scripts/train_dock_guide.py"
    echo "Training Guide Model (Binding Affinity)"
    echo "Config: $CONFIG"
    python "$SCRIPT" "$CONFIG"
    ;;
  
  guide_qed)
    CONFIG="$SCRIPT_DIR/configs/training_dock_guide_qed.yml"
    SCRIPT="$SCRIPT_DIR/scripts/train_dock_guide.py"
    echo "Training Guide Model (QED)"
    echo "Config: $CONFIG"
    python "$SCRIPT" "$CONFIG"
    ;;
  
  guide_sa)
    CONFIG="$SCRIPT_DIR/configs/training_dock_guide_sa.yml"
    SCRIPT="$SCRIPT_DIR/scripts/train_dock_guide.py"
    echo "Training Guide Model (SA)"
    echo "Config: $CONFIG"
    python "$SCRIPT" "$CONFIG"
    ;;
  
  *)
    echo "Unknown job type: $JOB_TYPE"
    echo "Valid options: diffusion, guide_ba, guide_qed, guide_sa"
    exit 1
    ;;
esac

EXIT_CODE=$?
echo ""
echo "Finished: $(date) (exit code: $EXIT_CODE)"

# Retry on failure
if [ $EXIT_CODE -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
  echo "Training failed. Resubmitting (retry $((RETRY_COUNT+1))/$MAX_RETRIES)..."
  sbatch "$SCRIPT_DIR/tagmol_train_slurm.sh" "$JOB_TYPE" $((RETRY_COUNT+1))
  exit 0
fi

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Training completed successfully!"
  echo "Checkpoints saved to: logs/"
fi

exit $EXIT_CODE
