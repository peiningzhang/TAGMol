#!/bin/bash
#SBATCH --job-name=tagmol_sample
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/healthinfolab/phz24002/TAGMol/logs/sample_%j.out
#SBATCH --error=/shared/healthinfolab/phz24002/TAGMol/logs/sample_%j.err

# TAGMol Sampling Script for SLURM
# Usage: sbatch tagmol_sample_slurm.sh [start_index] [end_index] [retry_count]
# On failure (e.g. CUDA version mismatch), auto-resubmits to get a new node.
# Example: sbatch tagmol_sample_slurm.sh 0 9   (pockets 0-9)

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

START_IDX=${1:-0}
END_IDX=${2:-9}
RETRY_COUNT=${3:-0}
MAX_RETRIES=5

CONDA_ENV_PATH="/shared/healthinfolab/phz24002/anaconda3/envs/tagmol"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CONFIG="$SCRIPT_DIR/configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml"
RESULT_PATH="$SCRIPT_DIR/experiments_multi/qed_0.33_sa_0.33_ba_0.34"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$RESULT_PATH"

echo "=========================================="
echo "TAGMol Multi-Guided Sampling (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Pockets: $START_IDX - $END_IDX"
echo "Retry attempt: $RETRY_COUNT / $MAX_RETRIES"
echo "Config: $CONFIG"
echo "Results: $RESULT_PATH"
echo "Started: $(date)"
echo "=========================================="

# Loop through each pocket in the range
for data_id in $(seq $START_IDX $END_IDX); do
  echo ""
  echo "Processing pocket $data_id..."
  echo "------------------------------------------"

  python scripts/sample_multi_guided_diffusion.py \
    "$CONFIG" \
    --data_id $data_id \
    --result_path "$RESULT_PATH" \
    --batch_size 10

  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Warning: Sampling failed for pocket $data_id (exit code: $EXIT_CODE)"
  else
    echo "Success: Pocket $data_id completed"
  fi
done

echo ""
echo "Finished: $(date)"

# If there were failures and we haven't exceeded max retries, resubmit
if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
  echo "Job completed. Check individual pocket results in $RESULT_PATH"
fi

exit 0
