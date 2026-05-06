#!/bin/bash

# SCHEDULERS=("log_uniform" "arcsin" "edm" "edm1")
# SCHEDULERS=("log_uniform" "arcsin" "edm1")
# SCHEDULERS=("log_uniform" "edm1")
            # --cfg_strategy "ramp_up" \
            # CFG_SCALES=(0 0.5 1.0 2.0 3.0 5.0)
SCHEDULERS=("gen_arcsin")
sigma_pocket_scales=(0 0.1 0.3 0.5 1.0)
for SCHED in "${SCHEDULERS[@]}"
do
    for sigma_pocket in "${sigma_pocket_scales[@]}"
    do
        echo "=========================================================="
        echo "Running quick_evaluate: time_scheduler=$SCHED sigma_pocket=$sigma_pocket"
        echo "=========================================================="
        # fixed cfg_scale for this sweep; do not put comments after trailing '\' (breaks bash line continuation)
        python scripts/quick_evaluate.py \
            --config logs_diffusion/training_cfg_muon_2026_04_20__16_14_01/sampling.yml \
            --checkpoint logs_diffusion/training_cfg_muon_2026_04_20__16_14_01/checkpoints/354000.pt \
            --num_proteins 100 \
            --num_ligands_per_protein 10 \
            --docking_mode vina_score \
            --export_grouped_sdf \
            --run_genbench \
            --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
            --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
            --genbench_do_conf_analysis \
            --genbench_no_vina \
            --time_scheduler "$SCHED" \
            --cfg_scale 5.0 \
            --sigma_pocket "$sigma_pocket"
        echo "Done: time_scheduler=$SCHED sigma_pocket=$sigma_pocket"
        echo ""
    done
done
