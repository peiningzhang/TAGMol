# 2) + PocketVE backbone
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/362000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --cfg_scale 0.0 \
  --no_veda_noise_injection

# 3)  + noise injection
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/362000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --cfg_scale 0.0 

# 4) + gen-arcsin schedule
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/362000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --time_scheduler gen_arcsin \
  --cfg_scale 0.0

  # 4.5) + s=1
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/362000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --time_scheduler gen_arcsin \
  --cfg_scale 1.0

# 5) + CFG
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_09__02_09_04/checkpoints/362000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --time_scheduler gen_arcsin \
  --cfg_scale 5.0

# 6) + protein perturb.
python scripts/quick_evaluate.py \
  --config logs_diffusion/training_cfg_muon_2026_04_17__13_40_04/sampling.yml \
  --checkpoint logs_diffusion/training_cfg_muon_2026_04_17__13_40_04/checkpoints/354000.pt \
  --num_proteins 100 \
  --num_ligands_per_protein 10 \
  --docking_mode vina_score \
  --export_grouped_sdf \
  --run_genbench \
  --gb3d_dir /shared/healthinfolab/phz24002/genbench3d \
  --genbench_python /home/phz24002/anaconda3/envs/genbench3d/bin/python \
  --genbench_do_conf_analysis \
  --genbench_no_vina \
  --time_scheduler gen_arcsin \
  --cfg_scale 5.0
