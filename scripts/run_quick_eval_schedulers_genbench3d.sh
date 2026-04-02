#!/bin/bash

# SCHEDULERS=("log_uniform" "arcsin" "edm" "edm1")
# SCHEDULERS=("log_uniform" "arcsin" "edm1")
SCHEDULERS=("log_uniform" "edm1")
for SCHED in "${SCHEDULERS[@]}"
do
    echo "=========================================================="
    echo "Running quick_evaluate with time_scheduler: $SCHED"
    echo "=========================================================="
    python scripts/quick_evaluate.py \
        --config logs_diffusion/training_2026_03_17__15_30_20/sampling.yml \
        --checkpoint logs_diffusion/training_2026_03_17__15_30_20/checkpoints/last.pt \
        --num_proteins 100 \
        --num_ligands_per_protein 10 \
        --docking_mode vina_score \
        --export_grouped_sdf \
        --run_genbench \
        --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
        --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
        --genbench_do_conf_analysis \
        --genbench_no_vina \
        --time_scheduler "$SCHED"
    echo "Done with time_scheduler: $SCHED"
    echo ""
done
