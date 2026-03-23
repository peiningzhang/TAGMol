首先
source /shared/healthinfolab/phz24002/anaconda3/bin/activate
conda activate tagmol
cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH=":"$PYTHONPATH
不断运行python scripts/quick_evaluate.py   --config logs_diffusion/training_2026_03_17__15_30_20/sampling.yml   --checkpoint logs_diffusion/training_2026_03_17__15_30_20/checkpoints/last.pt   --num_proteins 10 --num_ligands_per_protein 10 --docking_mode vina_score --guide_checkpoint logs/training_dock_guide_veda_2026_03_20__10_30_47/checkpoints/last.pt --guide_scale_cord 0.1 --guide_scale_categ -10 --num_steps 100
你可以更改两个参数guide_scale_cord和guide_scale_categ，来控制生成coordinate和discrete category的guidance权重。我不确定这两者应该是正数还是负数，你可以尝试一下。初步认为，coordinate应该取正数，discrete category应该取负数。
你可以尝试约50次，不断调整参数和分析结果，将结果写到BA_guide_test.csv中