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
#SBATCH --constraint="a100"

# Unified sampling script from trained model (supports both basic and multi-guided)
# Usage: sbatch sample_trained_slurm.sh [mode] [start_id] [end_id]
#   mode: basic (default) | multi
#   start_id: starting data_id (default: 0)
#   end_id: ending data_id (default: 9)
#
# Examples:
#   sbatch sample_trained_slurm.sh basic 0 9      # Basic sampling (pockets 0-9)
#   sbatch sample_trained_slurm.sh multi 10 19    # Multi-guided sampling (pockets 10-19)

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

MODE=${1:-basic}
START_IDX=${2:-0}
END_IDX=${3:-9}

CONDA_ENV_PATH="/shared/healthinfolab/phz24002/anaconda3/envs/tagmol"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Your trained checkpoint
CHECKPOINT="$SCRIPT_DIR/logs_diffusion/training_2026_03_06__02_00_45/checkpoints/176000.pt"

mkdir -p "$SCRIPT_DIR/logs"

echo "=========================================="
echo "TAGMol Sampling from Trained Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Mode: $MODE (basic=backbone only, multi=with QED+SA+BA guides)"
echo "Checkpoint: $CHECKPOINT"
echo "Data range: $START_IDX - $END_IDX"
echo "Started: $(date)"
echo "=========================================="

# Set config and result path based on mode
if [ "$MODE" = "multi" ]; then
    CONFIG="$SCRIPT_DIR/configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml"
    RESULT_PATH="$SCRIPT_DIR/experiments_multi/trained_176000"
    SCRIPT="scripts/sample_multi_guided_diffusion.py"
    echo "Using multi-guided sampling (QED + SA + BA)"
    echo "Config: $CONFIG"
    echo "Guides: QED (0.33) + SA (0.33) + BA (0.34)"
else
    CONFIG="$SCRIPT_DIR/configs/sampling.yml"
    RESULT_PATH="$SCRIPT_DIR/experiments/trained_176000"
    SCRIPT="scripts/sample_diffusion.py"
    echo "Using basic sampling (backbone only)"
    echo "Config: $CONFIG"
fi

echo "Script: $SCRIPT"
echo "Results: $RESULT_PATH"
echo "=========================================="

mkdir -p "$RESULT_PATH"

# Sample each data point in the range
for data_id in $(seq $START_IDX $END_IDX); do
    echo ""
    echo "[$MODE] Processing data_id $data_id..."
    
    python "$SCRIPT" \
        "$CONFIG" \
        --data_id $data_id \
        --checkpoint "$CHECKPOINT" \
        --result_path "$RESULT_PATH" \
        --batch_size 100
    
    if [ $? -eq 0 ]; then
        echo "✓ data_id $data_id completed"
    else
        echo "✗ data_id $data_id failed"
    fi
done

echo ""
echo "Finished: $(date)"
echo "Results saved to: $RESULT_PATH"
echo ""
echo "To evaluate results, run:"
if [ "$MODE" = "multi" ]; then
    echo "  python scripts/evaluate_diffusion.py $RESULT_PATH --docking_mode vina_score --protein_root data/test_set"
else
    echo "  python scripts/evaluate_diffusion.py $RESULT_PATH --docking_mode vina_score --protein_root data/test_set"
fi
