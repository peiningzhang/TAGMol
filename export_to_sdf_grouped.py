import torch
from rdkit import Chem
import os
from collections import defaultdict

pt_file = "experiments/trained_176000/eval_results/metrics_-1.pt"
out_dir = "experiments/trained_176000/sdfs_grouped"
os.makedirs(out_dir, exist_ok=True)

data = torch.load(pt_file)
results = data.get('all_results', [])

grouped_mols = defaultdict(list)

for i, res in enumerate(results):
    mol = res.get('mol')
    if mol is None:
        continue
        
    # Extract protein name
    target_name = res.get('ligand_filename', f'target_{i}')
    if isinstance(target_name, str):
        if target_name.startswith('LIGAND_'):
            target_name = target_name.replace('LIGAND_', '')
        
        # We only want the base name of the target directory/pocket
        target_name_base = target_name.split('.')[0].replace('/', '_')
        
        # CrossDocked test set directories are usually the pocket names directly.
        # But in evaluate_for_pocket, it was given the full path sometimes.
        # Let's just group by whatever string we got up to the sdf/pdb extension.
    else:
        target_name_base = f'target_{i}'
    
    # Set properties
    if 'chem_results' in res and isinstance(res['chem_results'], dict):
        if 'qed' in res['chem_results']:
            mol.SetProp('QED', str(res['chem_results']['qed']))
        if 'sa' in res['chem_results']:
            mol.SetProp('SA', str(res['chem_results']['sa']))
            
    if 'vina' in res and res['vina']:
        try:
            val = res['vina'].get('score_only', [{}])[0].get('affinity', '')
            mol.SetProp('Vina', str(val))
        except:
            pass

    grouped_mols[target_name_base].append(mol)

count_files = 0
count_mols = 0
for target_name, mols in grouped_mols.items():
    sdf_path = os.path.join(out_dir, f"{target_name}_generated.sdf")
    writer = Chem.SDWriter(sdf_path)
    for m in mols:
        writer.write(m)
        count_mols += 1
    writer.close()
    count_files += 1

print(f"Successfully exported {count_mols} molecules into {count_files} grouped SDF files in {out_dir}!")
