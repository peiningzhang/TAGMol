# Weights & Biases (wandb) 配置指南

## 概述

训练脚本现在已集成 **wandb** 日志记录功能，可同时记录到 TensorBoard 和 wandb 云端。

## 已添加 wandb 的训练脚本

| 脚本 | 项目名 | 记录内容 |
|------|--------|----------|
| `train_diffusion.py` | `tagmol` | Loss, LR, Grad Norm, AUROC |
| `train_dock_guide.py` | `tagmol-guide` | Loss, LR, Grad Norm |

## 使用方法

### 1. 登录 wandb (首次使用)

```bash
source /shared/healthinfolab/phz24002/anaconda3/bin/activate tagmol
wandb login
```

按提示输入你的 wandb API key (从 https://wandb.ai/authorize 获取)

### 2. 运行训练 (自动记录到 wandb)

```bash
cd /shared/healthinfolab/phz24002/TAGMol
source /shared/healthinfolab/phz24002/anaconda3/bin/activate tagmol
export PYTHONPATH="."$PYTHONPATH

# Diffusion 模型训练
python scripts/train_diffusion.py configs/training.yml

# Guide 模型训练
python scripts/train_dock_guide.py configs/training_dock_guide.yml
```

或提交 SLURM 作业：

```bash
bash submit_tagmol_train.sh diffusion
```

### 3. 查看训练日志

- **本地**: TensorBoard logs 保存在 `logs_diffusion/[experiment]/`
- **云端**: 访问 https://wandb.ai/your-username/projects 查看

## 记录的训练指标

### Diffusion 模型 (`tagmol` 项目)

| 指标 | 说明 |
|------|------|
| `train/loss` | 训练总损失 |
| `train/loss_pos` | 位置损失 |
| `train/loss_v` | 原子类型损失 |
| `train/lr` | 学习率 |
| `train/grad` | 梯度范数 |
| `val/loss` | 验证损失 |
| `val/loss_pos` | 验证位置损失 |
| `val/loss_v` | 验证原子类型损失 |
| `val/atom_auroc` | 原子类型 AUROC |
| `best_val_loss` | 最佳验证损失 (Summary) |
| `best_iter` | 最佳迭代步数 (Summary) |

### Guide 模型 (`tagmol-guide` 项目)

| 指标 | 说明 |
|------|------|
| `train/loss` | 训练损失 |
| `train/lr` | 学习率 |
| `train/grad` | 梯度范数 |
| `val/loss` | 验证损失 |
| `best_val_loss` | 最佳验证损失 (Summary) |
| `best_iter` | 最佳迭代步数 (Summary) |

## 离线模式 (无网络环境)

如果集群没有网络访问：

```bash
export WANDB_MODE=offline
python scripts/train_diffusion.py configs/training.yml
```

训练结束后同步：

```bash
wandb sync logs/[experiment]/wandb/offline-run-*/
```

## 自定义配置

### 修改 wandb 项目名

编辑训练脚本中的 `wandb.init()` 调用：

```python
wandb.init(
    project="your-project-name",  # 修改这里
    name=f"{config_name}_{args.tag}" if args.tag else config_name,
    config=config,
    dir=log_dir,
    save_code=True
)
```

### 禁用 wandb

```bash
export WANDB_DISABLED=true
python scripts/train_diffusion.py configs/training.yml
```

## 常见问题

### Q: wandb login 后仍然提示未登录
A: 确保使用相同的环境：
```bash
source /shared/healthinfolab/phz24002/anaconda3/bin/activate tagmol
wandb login
```

### Q: 如何共享实验结果？
A: wandb 自动记录到云端，可通过链接分享。也可设置团队：
```python
wandb.init(project="tagmol", entity="your-team-name")
```

### Q: 如何对比多个实验？
A: 在 wandb web 界面中选择多个 run 进行对比，或使用：
```bash
wandb diff run_id1 run_id2
```
