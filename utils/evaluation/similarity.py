import numpy as np
from rdkit import Chem, DataStructs


def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mean_pairwise_tanimoto(mols):
    """Mean Tanimoto over all unordered pairs (RDKFingerprint, same as tanimoto_sim). None if <2 mols."""
    fps = [Chem.RDKFingerprint(m) for m in mols if m is not None]
    if len(fps) < 2:
        return None
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(sims))


def pocket_diversity(mols):
    """Diversity = 1 - mean pairwise RDK Tanimoto among generated ligands for one pocket."""
    mp = mean_pairwise_tanimoto(mols)
    if mp is None:
        return None
    return 1.0 - mp


def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim


def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)
