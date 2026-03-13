#!/usr/bin/env python
"""
统计训练时ligand位置的标准差

用法:
    python scripts/stat_ligand_pos_std.py configs/training.yml [--num_samples N] [--device cuda]
"""

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from torch_scatter import scatter_mean

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    """Center positions at protein center of mass"""
    if mode == 'none':
        offset = torch.zeros(1, 3, device=protein_pos.device)
        return protein_pos, ligand_pos, offset
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
        return protein_pos, ligand_pos, offset
    else:
        raise NotImplementedError(f"Unknown center_pos_mode: {mode}")


def compute_ligand_std(ligand_pos, batch_ligand):
    """
    计算每个ligand分子的位置标准差
    
    Args:
        ligand_pos: (N_atoms, 3) ligand原子位置
        batch_ligand: (N_atoms,) 每个原子所属的分子索引
        
    Returns:
        std_per_mol: (num_molecules,) 每个分子的标准差
    """
    # 计算每个分子的质心
    ligand_pos_mean_per_mol = scatter_mean(ligand_pos, batch_ligand, dim=0)  # (num_graphs, 3)
    
    # 将每个分子的原子位置相对于该分子质心中心化
    ligand_pos_centered = ligand_pos - ligand_pos_mean_per_mol[batch_ligand]  # (N_atoms, 3)
    
    # 计算每个分子的RMS（标准差）
    ligand_pos_std_per_mol = scatter_mean(
        ligand_pos_centered.norm(dim=-1)**2, 
        batch_ligand, 
        dim=0
    ).sqrt()  # (num_graphs,)
    
    return ligand_pos_std_per_mol


def compute_ligand_center_norm(ligand_pos, batch_ligand):
    """
    计算每个ligand分子质心相对于蛋白质质心的距离（绝对值）
    
    Args:
        ligand_pos: (N_atoms, 3) ligand原子位置（已相对于蛋白质质心中心化）
        batch_ligand: (N_atoms,) 每个原子所属的分子索引
        
    Returns:
        center_norm_per_mol: (num_molecules,) 每个分子质心到原点的距离
    """
    # 计算每个分子的质心（相对于蛋白质质心(0,0,0)的位置）
    ligand_center_per_mol = scatter_mean(ligand_pos, batch_ligand, dim=0)  # (num_graphs, 3)
    
    # 计算每个质心到原点的距离（绝对值）
    center_norm_per_mol = ligand_center_per_mol.norm(dim=-1)  # (num_graphs,)
    
    return center_norm_per_mol


def main():
    parser = argparse.ArgumentParser(description='统计训练时ligand位置的标准差')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='统计的样本数量（None表示使用全部训练数据）')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='数据加载器工作进程数')
    
    args = parser.parse_args()
    
    # 加载配置
    config = misc.load_config(args.config)
    logger = misc.get_logger('stat_ligand_std')
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 准备transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)
    
    # 加载数据集
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set = subsets['train']
    logger.info(f'Training set size: {len(train_set)}')
    
    # 限制样本数量
    if args.num_samples is not None and args.num_samples < len(train_set):
        train_set = torch.utils.data.Subset(train_set, range(args.num_samples))
        logger.info(f'Using subset: {len(train_set)} samples')
    
    # 创建数据加载器
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,  # 不需要shuffle，只是统计
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )
    
    # 获取center_pos_mode
    center_pos_mode = getattr(config.model, 'center_pos_mode', 'protein')
    logger.info(f'center_pos_mode: {center_pos_mode}')
    
    # 统计所有ligand的标准差和质心距离
    all_std_values = []
    all_center_norm_values = []
    num_molecules = 0
    
    logger.info('Computing ligand position statistics...')
    for batch in tqdm(train_loader, desc='Processing batches'):
        batch = batch.to(device)
        
        # 应用center_pos变换（与训练时一致）
        protein_pos, ligand_pos, offset = center_pos(
            batch.protein_pos,
            batch.ligand_pos,
            batch.protein_element_batch,
            batch.ligand_element_batch,
            mode=center_pos_mode
        )
        
        # 计算每个ligand分子的标准差
        std_per_mol = compute_ligand_std(ligand_pos, batch.ligand_element_batch)
        
        # 计算每个ligand分子质心相对于蛋白质质心的距离
        center_norm_per_mol = compute_ligand_center_norm(ligand_pos, batch.ligand_element_batch)
        
        # 收集结果
        all_std_values.append(std_per_mol.cpu().numpy())
        all_center_norm_values.append(center_norm_per_mol.cpu().numpy())
        num_molecules += len(std_per_mol)
    
    # 合并所有结果
    all_std_values = np.concatenate(all_std_values)
    all_center_norm_values = np.concatenate(all_center_norm_values)
    
    # 计算统计信息 - 标准差统计
    logger.info('\n' + '='*60)
    logger.info('Ligand Position Standard Deviation Statistics')
    logger.info('='*60)
    logger.info(f'Total molecules analyzed: {num_molecules}')
    logger.info(f'Mean std: {all_std_values.mean():.4f} Å')
    logger.info(f'Median std: {np.median(all_std_values):.4f} Å')
    logger.info(f'Std of std: {all_std_values.std():.4f} Å')
    logger.info(f'Min std: {all_std_values.min():.4f} Å')
    logger.info(f'Max std: {all_std_values.max():.4f} Å')
    logger.info('\nPercentiles:')
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        logger.info(f'  {p:2d}th percentile: {np.percentile(all_std_values, p):.4f} Å')
    logger.info('='*60)
    
    # 计算统计信息 - 质心距离统计
    logger.info('\n' + '='*60)
    logger.info('Ligand Center Distance Statistics (relative to protein center)')
    logger.info('='*60)
    logger.info(f'Total molecules analyzed: {num_molecules}')
    logger.info(f'Mean center distance: {all_center_norm_values.mean():.4f} Å')
    logger.info(f'Median center distance: {np.median(all_center_norm_values):.4f} Å')
    logger.info(f'Std of center distance: {all_center_norm_values.std():.4f} Å')
    logger.info(f'Min center distance: {all_center_norm_values.min():.4f} Å')
    logger.info(f'Max center distance: {all_center_norm_values.max():.4f} Å')
    logger.info('\nPercentiles:')
    for p in percentiles:
        logger.info(f'  {p:2d}th percentile: {np.percentile(all_center_norm_values, p):.4f} Å')
    logger.info('='*60)
    
    # 保存结果到文件
    output_file_std = 'ligand_pos_std_stats.npy'
    output_file_center = 'ligand_center_norm_stats.npy'
    np.save(output_file_std, all_std_values)
    np.save(output_file_center, all_center_norm_values)
    logger.info(f'\nSaved std values to: {output_file_std}')
    logger.info(f'Saved center distance values to: {output_file_center}')
    
    # 可选：绘制直方图
    try:
        import matplotlib.pyplot as plt
        
        # 绘制标准差直方图
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_std_values, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Ligand Position Standard Deviation (Å)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Ligand Position Std\n(Mean: {all_std_values.mean():.4f} Å, Median: {np.median(all_std_values):.4f} Å)')
        plt.grid(True, alpha=0.3)
        plt.axvline(all_std_values.mean(), color='r', linestyle='--', label=f'Mean: {all_std_values.mean():.4f}')
        plt.axvline(np.median(all_std_values), color='g', linestyle='--', label=f'Median: {np.median(all_std_values):.4f}')
        plt.legend()
        
        # 绘制质心距离直方图
        plt.subplot(1, 2, 2)
        plt.hist(all_center_norm_values, bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Ligand Center Distance from Protein Center (Å)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Ligand Center Distance\n(Mean: {all_center_norm_values.mean():.4f} Å, Median: {np.median(all_center_norm_values):.4f} Å)')
        plt.grid(True, alpha=0.3)
        plt.axvline(all_center_norm_values.mean(), color='r', linestyle='--', label=f'Mean: {all_center_norm_values.mean():.4f}')
        plt.axvline(np.median(all_center_norm_values), color='g', linestyle='--', label=f'Median: {np.median(all_center_norm_values):.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ligand_pos_statistics_histogram.png', dpi=150)
        logger.info('Saved histograms to: ligand_pos_statistics_histogram.png')
    except ImportError:
        logger.info('matplotlib not available, skipping histogram generation')


if __name__ == '__main__':
    main()
