import torch
from rdkit import Chem
import os

pt_file = "experiments/trained_176000/eval_results/metrics_-1.pt"
out_dir = "experiments/trained_176000/sdfs"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading data from {pt_file}...")
try:
    data = torch.load(pt_file)
except Exception as e:
    print(f"Error loading {pt_file}: {e}")
    exit(1)

results = data.get('all_results', [])
if not results:
    print("No 'all_results' found in the metrics file.")
    exit(1)

count = 0
for i, res in enumerate(results):
    mol = res.get('mol')
    if mol is None:
        continue
        
    # Extract protein name
    target_name = res.get('ligand_filename', f'target_{i}')
    if isinstance(target_name, str):
        if target_name.startswith('LIGAND_'):
            target_name = target_name.replace('LIGAND_', '')
        
        # We only want the base name of the target
        target_name = target_name.split('.')[0]
        # Replace slashes or unallowed chars in filename
        target_name = target_name.replace('/', '_')
    else:
        target_name = f'target_{i}'
    
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

    sdf_path = os.path.join(out_dir, f"{target_name}_gen_{i}.sdf")
    try:
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        count += 1
    except Exception as e:
        print(f"Failed to write config {i} to sdf: {e}")

print(f"✅ Successfully exported {count} SDF files to {out_dir}!")
