#!/bin/bash

# SCHEDULERS=("log_uniform" "arcsin" "edm" "edm1")
# SCHEDULERS=("log_uniform" "arcsin" "edm1")
# SCHEDULERS=("log_uniform" "edm1")
            # --cfg_strategy "ramp_up" \
SCHEDULERS=("edm1")
CFG_SCALES=(0 0.5 1.0 2.0 3.0 5.0)
for SCHED in "${SCHEDULERS[@]}"
do
    for CFG in "${CFG_SCALES[@]}"
    do
        echo "=========================================================="
        echo "Running quick_evaluate: time_scheduler=$SCHED cfg_scale=$CFG"
        echo "=========================================================="
        python scripts/quick_evaluate.py \
            --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
            --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/200000.pt \
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
            --cfg_scale "$CFG"
        echo "Done: time_scheduler=$SCHED cfg_scale=$CFG"
        echo ""
    done
done
