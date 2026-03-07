#!/bin/bash
# Quick sampling from trained model (local execution)
# Usage: bash sample_trained.sh [mode] [data_id]
#   mode: basic (default) | multi
#   data_id: protein id to sample (default: 0)
#
# Examples:
#   bash sample_trained.sh basic 0      # Basic sampling for data_id 0
#   bash sample_trained.sh multi 5      # Multi-guided sampling for data_id 5

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

MODE=${1:-basic}
DATA_ID=${2:-0}

CONDA_ENV_PATH="/shared/healthinfolab/phz24002/anaconda3/envs/tagmol"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Your trained checkpoint
CHECKPOINT="$SCRIPT_DIR/logs_diffusion/training_2026_03_06__02_00_45/checkpoints/176000.pt"

echo "=========================================="
echo "TAGMol Sampling from Trained Model"
echo "=========================================="

# Set config and result path based on mode
if [ "$MODE" = "multi" ]; then
    CONFIG="$SCRIPT_DIR/configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml"
    RESULT_PATH="$SCRIPT_DIR/experiments_multi/trained_176000"
    SCRIPT="scripts/sample_multi_guided_diffusion.py"
    echo "Mode: Multi-guided sampling (QED + SA + BA)"
else
    CONFIG="$SCRIPT_DIR/configs/sampling.yml"
    RESULT_PATH="$SCRIPT_DIR/experiments/trained_176000"
    SCRIPT="scripts/sample_diffusion.py"
    echo "Mode: Basic sampling (backbone only)"
fi

echo "Checkpoint: $CHECKPOINT"
echo "Data ID: $DATA_ID"
echo "Results: $RESULT_PATH"
echo "=========================================="

mkdir -p "$RESULT_PATH"

python "$SCRIPT" \
    "$CONFIG" \
    --data_id $DATA_ID \
    --checkpoint "$CHECKPOINT" \
    --result_path "$RESULT_PATH" \
    --batch_size 100

echo ""
if [ $? -eq 0 ]; then
    echo "✓ Sampling completed successfully!"
    echo "Results saved to: $RESULT_PATH/result_${DATA_ID}.pt"
else
    echo "✗ Sampling failed"
    exit 1
fi
