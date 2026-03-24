# SLURM CPU资源申请问题

## 问题：可能申请不到8个CPU

是的，**确实可能申请不到8个CPU**。以下是可能的原因和解决方案。

## 可能的原因

### 1. **GPU节点的CPU核心数不足** ⚠️

某些GPU节点可能只有：
- 4-6个CPU核心（较老的节点）
- 8个CPU核心（刚好够用）
- 16+个CPU核心（充足）

当使用 `--constraint="a100"` 时，可能匹配到CPU较少的A100节点。

### 2. **分区限制** ⚠️

`general-gpu` 分区可能有：
- 每个任务的最大CPU限制（例如最多6个）
- CPU/GPU比例限制（例如每个GPU最多4个CPU）

### 3. **资源竞争** ⚠️

- 其他作业占用了CPU
- 节点资源不足
- 需要等待资源释放

### 4. **SLURM行为** ⚠️

SLURM的行为取决于配置：
- **严格模式**：如果节点没有8个CPU，作业会一直等待或失败
- **宽松模式**：可能分配少于8个CPU（例如只有4个）

## 检查方法

### 方法1：运行检查脚本

```bash
bash scripts/check_slurm_resources.sh
```

这会显示：
- 分区的CPU限制
- 可用节点的CPU数量
- 当前资源使用情况

### 方法2：手动检查

```bash
# 查看分区信息
scontrol show partition general-gpu

# 查看A100节点的CPU配置
sinfo -p general-gpu -o "%N %c %m %G" | grep -i a100

# 查看特定节点的详细信息
scontrol show node <node_name> | grep CPU
```

### 方法3：测试资源请求

```bash
# 测试是否能申请到8个CPU
srun --partition=general-gpu --gres=gpu:1 --cpus-per-task=8 \
     --constraint="a100" --time=1:00:00 --test-only \
     echo "Test" 2>&1
```

## 解决方案

### 方案1：减少CPU请求（推荐）

如果8个CPU申请不到，可以减少到4-6个：

```bash
#SBATCH --cpus-per-task=4  # 或 6
```

然后相应调整 `num_workers`：
```yaml
train:
  num_workers: 4  # 匹配分配的CPU数
```

### 方案2：使用更灵活的资源请求

不指定 `--constraint`，让SLURM选择任何可用节点：

```bash
#SBATCH --cpus-per-task=8
# 移除或注释掉: #SBATCH --constraint="a100"
```

### 方案3：使用优先级队列

如果 `priority-gpu` 或 `priority-l40` 有更多资源：

```bash
#SBATCH --partition=priority-gpu
#SBATCH --cpus-per-task=8
```

### 方案4：动态调整num_workers

在训练脚本中根据实际分配的CPU数调整：

```python
import os
allocated_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
config.train.num_workers = min(allocated_cpus, config.train.num_workers)
```

## 当前脚本的改进

我已经修改了 `tagmol_train_slurm.sh`，添加了：

1. **资源检查**：显示实际分配的CPU数
2. **警告信息**：如果分配的CPU少于请求的，会显示警告

### 运行时会显示：

```
Allocated CPUs: 8  # 或实际分配的数量
Allocated Memory: 65536 MB
GPU: 0
```

如果只分配到4个CPU：
```
WARNING: Only 4 CPUs allocated (requested 8)
Consider reducing num_workers in config if data loading is slow
```

## 推荐配置

### 如果经常申请不到8个CPU：

**修改 `tagmol_train_slurm.sh`：**
```bash
#SBATCH --cpus-per-task=4  # 从8改为4
```

**修改 `configs/training.yml`：**
```yaml
train:
  num_workers: 4  # 匹配CPU数
```

### 如果8个CPU足够但偶尔申请不到：

保持当前配置，SLURM会：
- 等待直到有8个CPU可用
- 或者你可以手动重试

## 验证实际分配的CPU数

作业运行后，检查日志文件：
```bash
grep "Allocated CPUs" logs/train_*.out
```

或者在作业脚本中添加：
```bash
echo "Actual CPUs: $SLURM_CPUS_PER_TASK"
echo "Actual Memory: $SLURM_MEM_PER_NODE MB"
```

## 最佳实践

1. **先检查资源**：运行 `check_slurm_resources.sh` 了解可用资源
2. **保守请求**：如果经常申请不到，减少CPU请求
3. **匹配配置**：确保 `num_workers` 不超过分配的CPU数
4. **监控日志**：检查实际分配的CPU数，调整配置

## 常见情况

| 情况 | CPU请求 | 实际分配 | 建议 |
|------|---------|---------|------|
| 节点有16+核心 | 8 | 8 | ✅ 正常 |
| 节点只有4核心 | 8 | 4 | ⚠️ 减少请求到4 |
| 分区限制6个 | 8 | 6 | ⚠️ 减少请求到6 |
| 资源竞争 | 8 | 等待/失败 | ⚠️ 使用优先级队列 |
