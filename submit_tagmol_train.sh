#!/bin/bash
# Submit TAGMol training job to general-gpu
# Usage: bash submit_tagmol_train.sh [diffusion|guide_ba|guide_qed|guide_sa]

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

mkdir -p logs

JOB_TYPE=${1:-diffusion}

case $JOB_TYPE in
  diffusion)
    echo "Submitting Diffusion model training job..."
    sbatch --job-name=tagmol_train_diff tagmol_train_slurm.sh diffusion
    ;;
  guide_ba)
    echo "Submitting Guide (Binding Affinity) training job..."
    sbatch --job-name=tagmol_train_ba tagmol_train_slurm.sh guide_ba
    ;;
  guide_qed)
    echo "Submitting Guide (QED) training job..."
    sbatch --job-name=tagmol_train_qed tagmol_train_slurm.sh guide_qed
    ;;
  guide_sa)
    echo "Submitting Guide (SA) training job..."
    sbatch --job-name=tagmol_train_sa tagmol_train_slurm.sh guide_sa
    ;;
  *)
    echo "Unknown job type: $JOB_TYPE"
    echo "Usage: bash submit_tagmol_train.sh [diffusion|guide_ba|guide_qed|guide_sa]"
    exit 1
    ;;
esac

echo ""
echo "Job submitted. Check status with: squeue -u \$USER"
echo "Logs will be saved to: logs/"
