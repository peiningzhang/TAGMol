import torch
import glob
from collections import Counter
from rdkit import Chem
from utils.evaluation import analyze

files = glob.glob('*/eval_results/metrics_*.pt') + glob.glob('*/*/eval_results/metrics_*.pt')
f = sorted(files)[0]
print("Loading:", f)
data = torch.load(f)
results = data.get('all_results', [])

atoms_unstable = []
bonds_count = Counter()
total_atoms = 0

for res in results[:20]:
    mol = res['mol']
    if mol is None: continue
    
    # We can calculate number of bonds via RDKit properties
    # Let's get them from pred_pos and pred_v since stability is defined over 3D euclidean distance in `analyze.py`
    pos = res.get('pred_pos')
    pred_v = res.get('pred_v')
    
    if pos is None or pred_v is None:
        continue
        
    from utils import transforms
    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode='add_aromatic')
    
    # Pass to check_stability
    molecule_stable, nr_stable_bonds, n_atoms, nr_bonds = analyze.check_stability(pos, pred_atom_type, return_nr_bonds=True)
    total_atoms += n_atoms
    
    for i, a in enumerate(pred_atom_type):
        sym = analyze.atom_decoder[a]
        allowed = analyze.allowed_bonds[sym]
        actual = nr_bonds[i]
        is_stable = (allowed >= actual > 0)
        if not is_stable:
            atoms_unstable.append((sym, allowed, actual))
        bonds_count[(sym, actual)] += 1

print("Total atoms checked:", total_atoms)
print("Total unstable atoms:", len(atoms_unstable))
print("Sample unstable:", atoms_unstable[:20])
print("Sample bonds count:", bonds_count.most_common(20))
