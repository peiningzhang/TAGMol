#!/usr/bin/env python
"""
诊断 val/atom_auroc 不增长的问题

用法:
    python scripts/diagnose_atom_auroc.py configs/training.yml --checkpoint <checkpoint_path>
"""

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from collections import Counter

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from scripts.train_diffusion import get_auroc


def diagnose_auroc(config, checkpoint_path=None, device='cuda', num_samples=100):
    """诊断AUROC问题"""
    
    print("="*80)
    print("Atom AUROC Diagnosis")
    print("="*80)
    
    # 加载数据
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    val_set = subsets['test']
    
    # 限制样本数量
    if num_samples < len(val_set):
        val_set = torch.utils.data.Subset(val_set, range(num_samples))
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=['ligand_nbh_list']
    )
    
    # 加载模型
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(device)
    
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Using randomly initialized model")
    
    model.eval()
    
    # 1. 检查atom type分布
    print("\n1. Atom Type Distribution:")
    all_atom_types = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            all_atom_types.extend(batch.ligand_atom_feature_full.cpu().numpy().tolist())
    
    atom_type_counts = Counter(all_atom_types)
    total = len(all_atom_types)
    print(f"   Total atoms: {total}")
    print(f"   Unique atom types: {len(atom_type_counts)}")
    print("\n   Top 10 atom types:")
    for atom_type, count in atom_type_counts.most_common(10):
        percentage = count / total * 100
        print(f"     Type {atom_type}: {count} ({percentage:.1f}%)")
    
    # 2. 检查不同time_step的AUROC
    print("\n2. AUROC by Timestep:")
    num_val_timesteps = 10
    timesteps = np.linspace(0, model.num_timesteps - 1, num_val_timesteps).astype(int)
    
    auroc_by_timestep = {}
    loss_v_by_timestep = {}
    
    with torch.no_grad():
        for t in timesteps:
            all_pred_v = []
            all_true_v = []
            sum_loss_v = 0
            sum_n = 0
            
            for batch in val_loader:
                batch = batch.to(device)
                batch_size = batch.num_graphs
                time_step = torch.tensor([t] * batch_size).to(device)
                
                results = model.get_diffusion_loss(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                    time_step=time_step
                )
                
                all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                sum_loss_v += float(results['loss_v']) * batch_size
                sum_n += batch_size
            
            # 计算AUROC
            pred_v = np.concatenate(all_pred_v, axis=0)
            true_v = np.concatenate(all_true_v)
            auroc = get_auroc(true_v, pred_v, config.data.transform.ligand_atom_mode)
            avg_loss_v = sum_loss_v / sum_n
            
            auroc_by_timestep[t] = auroc
            loss_v_by_timestep[t] = avg_loss_v
            
            print(f"   t={t:4d}: AUROC={auroc:.4f}, Loss_v={avg_loss_v:.6f}")
    
    # 3. 分析
    print("\n3. Analysis:")
    clean_auroc = auroc_by_timestep[0]
    noisy_timesteps = [t for t in timesteps if t > 0]
    noisy_aurocs = [auroc_by_timestep[t] for t in noisy_timesteps]
    avg_noisy_auroc = np.mean(noisy_aurocs)
    
    print(f"   Clean data (t=0) AUROC: {clean_auroc:.4f}")
    print(f"   Noisy data (t>0) average AUROC: {avg_noisy_auroc:.4f}")
    print(f"   Difference: {clean_auroc - avg_noisy_auroc:.4f}")
    
    if clean_auroc > 0.95:
        print("\n   ⚠️  WARNING: Clean data AUROC is very high (>0.95)")
        print("      This suggests the model is evaluated on nearly clean data at t=0")
        print("      The high AUROC may not reflect the model's true learning progress")
    
    if avg_noisy_auroc < clean_auroc - 0.1:
        print("\n   ⚠️  WARNING: Large gap between clean and noisy AUROC")
        print("      This suggests the model struggles with noisy data")
        print("      Consider focusing on noisy timesteps during validation")
    
    # 4. 建议
    print("\n4. Recommendations:")
    if clean_auroc > avg_noisy_auroc + 0.1:
        print("   ✅ Exclude t=0 from AUROC calculation (or report separately)")
        print("   ✅ Focus on noisy timesteps (t > 0) for evaluation")
        print("   ✅ Use weighted average, giving more weight to noisy timesteps")
    
    if len(atom_type_counts) < 5:
        print("   ⚠️  Very few atom types in validation set")
        print("      High AUROC might be due to class imbalance")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='诊断atom AUROC问题')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint路径（可选）')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--num_samples', type=int, default=100, help='使用的验证样本数量')
    
    args = parser.parse_args()
    config = misc.load_config(args.config)
    
    diagnose_auroc(config, args.checkpoint, args.device, args.num_samples)
