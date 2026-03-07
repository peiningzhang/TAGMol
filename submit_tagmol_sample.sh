#!/bin/bash
# Submit TAGMol sampling jobs to general-gpu (10 pockets per job, 10 jobs total)
# Usage: bash submit_tagmol_sample.sh

SCRIPT_DIR="/shared/healthinfolab/phz24002/TAGMol"
cd "$SCRIPT_DIR" || exit 1

mkdir -p logs

# 100 pockets, 10 per job -> 10 jobs
# Groups: 0-9, 10-19, 20-29, ..., 90-99
for i in $(seq 0 9); do
  START=$((i * 10))
  END=$((START + 9))
  echo "Submitting job for pockets $START-$END..."
  sbatch --job-name=tagmol_s${START}-${END} tagmol_sample_slurm.sh $START $END
done

echo ""
echo "Submitted 10 jobs. Check status with: squeue -u \$USER"
echo "Results will be saved to: experiments_multi/qed_0.33_sa_0.33_ba_0.34/"
