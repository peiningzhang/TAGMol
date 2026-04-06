#!/usr/bin/env python
"""
分析训练时GPU利用率低的原因

用法:
    python scripts/analyze_gpu_utilization.py configs/training.yml
"""

import argparse
import torch
import time
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


def analyze_training_bottlenecks(config, device='cuda'):
    """分析训练瓶颈"""
    
    print("="*80)
    print("GPU Utilization Analysis")
    print("="*80)
    
    # 1. 检查配置参数
    print("\n1. Configuration Analysis:")
    print(f"   Batch size: {config.train.batch_size}")
    print(f"   Num workers: {config.train.num_workers}")
    print(f"   Gradient accumulation: {config.train.n_acc_batch}")
    print(f"   Effective batch size: {config.train.batch_size * config.train.n_acc_batch}")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n2. GPU Information:")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("\n2. GPU: Not available")
        return
    
    # 2. 加载数据和模型
    print("\n3. Loading Dataset and Model...")
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    train_set = subsets['train']
    
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )
    
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    
    num_params = misc.count_parameters(model) / 1e6
    print(f"   Model parameters: {num_params:.2f} M")
    
    # 3. 分析数据加载时间
    print("\n4. Data Loading Analysis:")
    data_load_times = []
    batch_sizes = []
    num_atoms = []
    
    torch.cuda.synchronize()
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只测试前10个batch
            break
        batch_load_time = time.time()
        batch = batch.to(device)
        torch.cuda.synchronize()
        data_load_times.append(time.time() - batch_load_time)
        batch_sizes.append(batch.num_graphs)
        num_atoms.append(batch.protein_pos.shape[0] + batch.ligand_pos.shape[0])
    
    total_time = time.time() - start
    avg_load_time = np.mean(data_load_times)
    
    print(f"   Average data loading time per batch: {avg_load_time*1000:.2f} ms")
    print(f"   Average batch size: {np.mean(batch_sizes):.2f}")
    print(f"   Average atoms per batch: {np.mean(num_atoms):.0f}")
    print(f"   Data loading throughput: {len(data_load_times)/total_time:.2f} batches/sec")
    
    # 4. 分析前向传播时间
    print("\n5. Forward Pass Analysis:")
    model.train()
    forward_times = []
    backward_times = []
    
    batch = next(iter(train_loader)).to(device)

    # Warmup
    for _ in range(3):
        results = model.get_diffusion_loss(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch
        )
        loss = results['loss']
        loss.backward()
    
    # 实际测试
    for _ in range(10):
        torch.cuda.synchronize()
        forward_start = time.time()
        
        results = model.get_diffusion_loss(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch
        )
        loss = results['loss']
        torch.cuda.synchronize()
        forward_times.append(time.time() - forward_start)
        
        torch.cuda.synchronize()
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - backward_start)
    
    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)
    total_time_per_batch = avg_forward + avg_backward + avg_load_time
    
    print(f"   Average forward pass time: {avg_forward*1000:.2f} ms")
    print(f"   Average backward pass time: {avg_backward*1000:.2f} ms")
    print(f"   Total time per batch: {total_time_per_batch*1000:.2f} ms")
    print(f"   Training throughput: {1/total_time_per_batch:.2f} batches/sec")
    
    # 5. GPU利用率分析
    print("\n6. GPU Utilization Analysis:")
    gpu_utilization = (avg_forward + avg_backward) / total_time_per_batch * 100
    cpu_utilization = avg_load_time / total_time_per_batch * 100
    
    print(f"   GPU compute time ratio: {gpu_utilization:.1f}%")
    print(f"   Data loading time ratio: {cpu_utilization:.1f}%")
    print(f"   Idle time ratio: {100 - gpu_utilization - cpu_utilization:.1f}%")
    
    # 6. 内存使用
    print("\n7. Memory Usage:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Total: {total:.2f} GB")
        print(f"   Utilization: {reserved/total*100:.1f}%")
    
    # 7. 建议
    print("\n8. Recommendations:")
    recommendations = []
    
    if config.train.batch_size < 8:
        recommendations.append(f"   ⚠️  Batch size ({config.train.batch_size}) is very small. Consider increasing to 8-16 or higher.")
    
    if config.train.num_workers < 8:
        recommendations.append(f"   ⚠️  Num workers ({config.train.num_workers}) may be insufficient. Consider increasing to 8-16.")
    
    if cpu_utilization > 30:
        recommendations.append(f"   ⚠️  Data loading takes {cpu_utilization:.1f}% of time. Increase num_workers or use pin_memory=True.")
    
    if gpu_utilization < 50:
        recommendations.append(f"   ⚠️  GPU utilization is only {gpu_utilization:.1f}%. Increase batch_size or use gradient accumulation.")
    
    if config.train.n_acc_batch == 1 and config.train.batch_size < 8:
        recommendations.append(f"   💡  Consider using gradient accumulation (n_acc_batch > 1) to increase effective batch size.")
    
    if len(recommendations) == 0:
        print("   ✅ Configuration looks good!")
    else:
        for rec in recommendations:
            print(rec)
    
    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析GPU利用率')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    config = misc.load_config(args.config)
    
    analyze_training_bottlenecks(config, args.device)
