#!/bin/bash
# Submit sampling jobs from trained model (138000.pt)
# Usage: bash submit_sample_trained.sh [mode]
#   mode: basic (default) | multi
#
# Examples:
#   bash submit_sample_trained.sh basic    # Submit basic sampling jobs (10 jobs)
#   bash submit_sample_trained.sh multi    # Submit multi-guided sampling jobs (10 jobs)

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

MODE=${1:-basic}
mkdir -p logs

echo "=========================================="
echo "Submitting TAGMol Sampling Jobs"
echo "=========================================="

if [ "$MODE" = "multi" ]; then
    echo "Mode: Multi-guided sampling (QED + SA + BA)"
    echo "Config: configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml"
    echo "Results: experiments_multi/trained_176000/"
    JOB_PREFIX="multi"
else
    echo "Mode: Basic sampling (backbone only)"
    echo "Config: configs/sampling.yml"
    echo "Results: experiments/trained_176000/"
    JOB_PREFIX="basic"
fi

echo "Checkpoint: logs_diffusion/training_2026_03_06__02_00_45/checkpoints/138000.pt"
echo "Jobs: 10 groups (10 proteins per job)"
echo "=========================================="
echo ""

# Submit 10 jobs (100 pockets, 10 per job)
for i in $(seq 0 9); do
  START=$((i * 10))
  END=$((START + 9))
  echo "Submitting job for pockets $START-$END (${JOB_PREFIX}_176k_${START}-${END})..."
  sbatch --job-name=${JOB_PREFIX}_176k_${START}-${END} sample_trained_slurm.sh $MODE $START $END
done

echo ""
echo "=========================================="
echo "Submitted 10 jobs!"
echo "=========================================="
echo "Check status: squeue -u \$USER"
echo "View logs:   tail -f logs/sample_*.out"
echo ""
if [ "$MODE" = "multi" ]; then
    echo "After completion, evaluate with:"
    echo "  python scripts/evaluate_diffusion.py experiments_multi/trained_176000 --docking_mode vina_score --protein_root data/test_set"
else
    echo "After completion, evaluate with:"
    echo "  python scripts/evaluate_diffusion.py experiments/trained_176000 --docking_mode vina_score --protein_root data/test_set"
fi
