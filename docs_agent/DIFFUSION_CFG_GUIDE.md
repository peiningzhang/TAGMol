# Diffusion CFG Guide

## 目标

这份说明记录 TAGMol 的 diffusion 条件化改动，重点是 classifier-free guidance（CFG）的训练和采样入口。

当前实现采用：
- `vina / QED / SA` 作为条件
- 每个属性独立分桶
- `null index` 作为无条件分支
- 训练时随机丢弃条件
- 采样时做 `cond / uncond` 双前向并按 CFG 合成

---

## 数据前提

条件只对带属性的数据集有效。

可用数据链路：
- `pl`: 基础 pair 数据，只含蛋白/配体对
- `pl_dock_guide`: 同一批 pair，额外带 `vina_dock / qed / sa / brenk_pass`

相关实现：
- `datasets/pl_pair_dataset.py`
- `datasets/pl_pair_dock_guide_dataset.py`

如果要做条件训练，应该使用 `pl_dock_guide` 对应的数据和 split。

---

## 条件分桶

条件分桶由训练集统计得到，并保存成 YAML。

生成脚本：
```bash
python scripts/data_preparation/build_condition_bins.py \
  configs/training_dock_guide.yml \
  --output /shared/healthinfolab/phz24002/AliDiff/data/condition_bins.yml
```

分桶规则：
- `vina`：越小越好，内部会反向后再分桶
- `QED`：越大越好
- `SA`：越大越好
- 采用 5 桶等频切分

---

## 训练配置

模型侧默认不启用 condition，保持旧逻辑不变。

开启条件训练时，至少需要：
```yaml
data:
  name: pl_dock_guide
  path: /shared/healthinfolab/phz24002/AliDiff/data/crossdocked_v1.1_rmsd1.0_pocket10
  split: /shared/healthinfolab/phz24002/AliDiff/data/crossdocked_pocket10_pose_split_dock_guide.pt
  index_path: /shared/healthinfolab/phz24002/AliDiff/data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl
  transform:
    condition_bins_path: /shared/healthinfolab/phz24002/AliDiff/data/condition_bins.yml

model:
  use_condition: true
  condition_dropout: 0.2
  condition_bins: 5
  condition_emb_dim: 8
```

当前代码里：
- `use_condition=false` 时，条件分支完全不生效
- `condition_dropout` 只在训练时起作用
- `null index` 是条件字典里的最后一档，代表无条件

---

## CFG 采样

采样入口支持 `cfg_scale`。

默认值：
- `cfg_scale = 1.0`，等价于不启用 CFG

启用 CFG 时：
- 模型会做一次 conditional forward
- 再做一次 unconditional forward
- 最终按 `uncond + scale * (cond - uncond)` 合成

命令行入口：
```bash
python scripts/sample_diffusion.py configs/sampling.yml --cfg_scale 3.0
```

如果 checkpoint 里带了 `condition_bins_path`，采样脚本会自动加载条件特征。

---

## 实现细节

关键实现点：
- 条件 bin 通过 embedding 编码
- `null index` 对应单独的 embedding 行
- `condition_dropout` 时不是置零向量，而是把 bin 替换成 `null index`
- `null index` 与普通 0..4 桶不冲突

对应代码：
- `models/molopt_score_model.py`
- `scripts/train_diffusion.py`
- `scripts/sample_diffusion.py`
- `utils/transforms.py`

---

## 验证

先确认 `pl` 和 `pl_dock_guide` 的 pair 集合一致，再做条件训练。

验证脚本：
```bash
python scripts/data_preparation/validate_dock_guide_indices.py \
  --base_lmdb ../AliDiff/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
  --guide_lmdb ../AliDiff/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_dock_guide_final.lmdb \
  --full_compare
```

---

## 默认行为

如果不改配置：
- 训练行为不变
- 采样行为不变
- CFG 不会自动启用

只有显式打开 `use_condition` 并提供 `condition_bins_path`，才会进入条件路径。

