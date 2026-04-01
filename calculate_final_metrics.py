import json
import numpy as np
import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def calculate_diversity(mol_list):
    if len(mol_list) < 2:
        return 0.0
    
    fps = []
    for mol in mol_list:
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
    
    if len(fps) < 2:
        return 0.0
    
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
            
    # Diversity = 1 - average pairwise similarity
    return 1 - np.mean(similarities)

# Paths
json_path = "/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped/genbench_results/all_results_aggregated.json"
sdf_dir = "/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped"

# 1. Load Vina metrics from JSON
with open(json_path, 'r') as f:
    data = json.load(f)

total_mols = 0
high_affinity_count = 0
vina_scores = []

for pocket_res in data:
    rel_scores = pocket_res.get('Relative Min Vina score', [])
    abs_scores = pocket_res.get('Minimized Vina score', [])
    
    for score in rel_scores:
        if score is None or (isinstance(score, float) and np.isnan(score)): continue
        total_mols += 1
        if score < 0:
            high_affinity_count += 1
            
    vina_scores.extend([s for s in abs_scores if s is not None and not (isinstance(s, float) and np.isnan(s))])

high_affinity_rate = (high_affinity_count / total_mols) * 100 if total_mols > 0 else 0

# 2. Calculate Diversity from SDF files
# We need to read the molecules back to compute fingerprints
sdf_files = glob.glob(os.path.join(sdf_dir, "*_generated.sdf"))
pocket_diversities = []

print(f"Calculating diversity across {len(sdf_files)} pockets...")
for sdf in sdf_files:
    suppl = Chem.SDMolSupplier(sdf)
    mols = [m for m in suppl if m is not None]
    if len(mols) >= 2:
        div = calculate_diversity(mols)
        pocket_diversities.append(div)

mean_diversity = np.mean(pocket_diversities) if pocket_diversities else 0

# 3. Output Report
print("\n" + "="*40)
print("       TAGMol EVALUATION REPORT")
print("="*40)
print(f"Directory: {os.path.dirname(os.path.dirname(sdf_dir))}")
print(f"Total Molecules:     {total_mols}")
print(f"Mean Min Vina Score: {np.mean(vina_scores):.4f}")
print("-" * 40)
print(f"High Affinity Rate:  {high_affinity_rate:.2f}%")
print(f"Diversity:           {mean_diversity:.4f}")
print("="*40)

# Also save to a summary file for future reference
summary_path = os.path.join(os.path.dirname(json_path), "final_metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"High Affinity Rate: {high_affinity_rate:.2f}%\n")
    f.write(f"Diversity:          {mean_diversity:.4f}\n")
    f.write(f"Mean Vina Score:    {np.mean(vina_scores):.4f}\n")
    f.write(f"Total Samples:      {total_mols}\n")

print(f"\nFinal metrics summary saved to: {summary_path}")
