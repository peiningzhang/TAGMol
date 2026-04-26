#!/usr/bin/env bash
set -eo pipefail

source /shared/healthinfolab/phz24002/anaconda3/bin/activate
conda activate tagmol

cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH=":${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=2 python /shared/healthinfolab/phz24002/TAGMol/scripts/evaluate_diffusion.py results/ \
  --eval_step -1 \
  --eval_num_examples 100 \
  --docking_mode vina_score \
  --protein_root /shared/healthinfolab/phz24002/TAGMol/data/test_set \
  --save True >> results/evalation.log 2>&1

python /shared/healthinfolab/phz24002/TAGMol/export_to_sdf_grouped.py results/ --eval-step -1 >> results/evalation.log 2>&1
python /shared/healthinfolab/phz24002/TAGMol/run_genbench_eval.py --do_conf_analysis --no_vina --gen_dir results/sdfs_grouped >> results/evalation.log 2>&1
python /shared/healthinfolab/phz24002/TAGMol/get_genbench_report.py results/sdfs_grouped/genbench_results >> results/evalation.log 2>&1
