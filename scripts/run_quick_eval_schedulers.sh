#!/bin/bash

SCHEDULERS=("log_uniform" "arcsin" "edm" "edm1")

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
        --time_scheduler $SCHED
    echo "Done with time_scheduler: $SCHED"
    echo ""
done
