# 断点续训功能使用说明

## 功能说明

训练脚本现在支持从checkpoint恢复训练，可以继续之前的训练进度。

## 使用方法

### 1. 基本用法

```bash
python scripts/train_diffusion.py configs/training.yml --resume <checkpoint_path>
```

### 2. 使用 last.pt（推荐）

如果训练中断，可以使用最新的 `last.pt` 恢复：

```bash
python scripts/train_diffusion.py configs/training.yml \
    --resume logs_diffusion/training_YYYY_MM_DD__HH_MM_SS/checkpoints/last.pt
```

### 3. 使用特定iteration的checkpoint

```bash
python scripts/train_diffusion.py configs/training.yml \
    --resume logs_diffusion/training_YYYY_MM_DD__HH_MM_SS/checkpoints/20000.pt
```

### 4. 在SLURM脚本中使用

修改 `tagmol_train_slurm.sh` 或直接传递参数：

```bash
sbatch --job-name=tagmol_train_resume tagmol_train_slurm.sh diffusion \
    --resume logs_diffusion/training_YYYY_MM_DD__HH_MM_SS/checkpoints/last.pt
```

## 恢复的内容

断点续训会恢复以下内容：

1. **模型权重** (`model.state_dict()`)
   - 所有模型参数

2. **优化器状态** (`optimizer.state_dict()`)
   - Adam的momentum和二阶矩估计
   - 学习率等优化器参数

3. **调度器状态** (`scheduler.state_dict()`)
   - 学习率调度器的状态

4. **训练进度**
   - `iteration`: 从保存的iteration + 1开始
   - `best_loss`: 之前的最佳验证损失
   - `best_iter`: 之前的最佳iteration

## 注意事项

### ⚠️ 重要提示

1. **配置文件**
   - 使用命令行提供的配置文件，而不是checkpoint中的配置
   - 如果需要使用checkpoint的配置，需要手动指定

2. **数据路径**
   - 确保数据路径仍然有效
   - 如果数据路径改变，需要更新配置文件

3. **模型结构**
   - checkpoint中的模型结构必须与当前代码一致
   - 如果模型结构改变，可能无法加载

4. **wandb**
   - wandb会创建新的run，不会恢复之前的run
   - 如果需要继续之前的wandb run，需要手动设置run ID

## 示例场景

### 场景1：训练被中断

```bash
# 训练被Ctrl+C中断或SLURM任务超时
# 找到最新的checkpoint
ls -lt logs_diffusion/*/checkpoints/last.pt

# 恢复训练
python scripts/train_diffusion.py configs/training.yml \
    --resume logs_diffusion/training_2026_03_13__00_51_20/checkpoints/last.pt
```

### 场景2：从最佳模型继续训练

```bash
# 找到最佳模型的checkpoint（验证损失最低的iteration）
python scripts/train_diffusion.py configs/training.yml \
    --resume logs_diffusion/training_2026_03_13__00_51_20/checkpoints/20000.pt
```

### 场景3：修改配置后继续训练

```bash
# 修改了batch_size或其他训练参数，想从之前的模型继续
python scripts/train_diffusion.py configs/training_new.yml \
    --resume logs_diffusion/training_2026_03_13__00_51_20/checkpoints/last.pt
```

## Checkpoint文件结构

每个checkpoint文件包含：

```python
{
    'config': config,              # 训练时的配置
    'model': model.state_dict(),   # 模型权重
    'optimizer': optimizer.state_dict(),  # 优化器状态
    'scheduler': scheduler.state_dict(),   # 调度器状态
    'iteration': it,               # 当前iteration
    'best_loss': best_loss,        # 最佳验证损失
    'best_iter': best_iter,        # 最佳iteration
}
```

## 验证恢复是否成功

恢复训练时，日志会显示：

```
Resuming training from checkpoint: logs_diffusion/.../checkpoints/last.pt
Loaded model state
Loaded optimizer state
Loaded scheduler state
Resuming from iteration: 12345
Previous best loss: 0.123456
Previous best iteration: 12000
Checkpoint loaded successfully!
```

## 故障排除

### 问题1：找不到checkpoint文件

```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**解决**：检查路径是否正确，使用绝对路径或相对路径。

### 问题2：模型结构不匹配

```
RuntimeError: Error(s) in loading state_dict
```

**解决**：模型结构已改变，无法直接加载。可能需要：
- 使用 `strict=False` 加载（代码中已处理）
- 或重新训练

### 问题3：CUDA out of memory

**解决**：恢复训练时batch_size可能太大，减少batch_size：
```bash
python scripts/train_diffusion.py configs/training.yml \
    --resume <checkpoint> --batch_size 8
```

## 最佳实践

1. **定期保存checkpoint**
   - `last.pt` 每个iteration都保存（已实现）
   - 最佳模型也会保存（验证损失改善时）

2. **使用绝对路径**
   - 避免路径问题

3. **检查checkpoint完整性**
   ```python
   import torch
   ckpt = torch.load('checkpoint.pt')
   print(ckpt.keys())  # 应该包含 'model', 'optimizer', 'scheduler', 'iteration'
   ```

4. **备份重要checkpoint**
   - 定期备份最佳模型
