# TAGMol 环境设置与评估指南

## ✅ 环境状态 - 全部就绪

| 项目 | 状态 |
|------|------|
| Conda 环境 (`tagmol`) | ✅ Python 3.8.17 + PyTorch 1.13.1 + CUDA 11.6 |
| PyTorch Geometric 2.2.0 | ✅ 已安装 |
| RDKit, OpenBabel, TensorBoard 等 | ✅ 已安装 |
| Vina Docking 工具 | ✅ meeko, vina, pdb2pqr, AutoDockTools_py3 |
| 测试数据软链接 | ✅ `data/test_set` → `../AliDiff/data/test_set` |
| **权重文件** | ✅ **全部已就位** |

### 权重文件清单（已就绪）

```
TAGMol/
├── pretrained_models/
│   ├── pretrained_diffusion.pt          ✅ (32M) - Backbone 扩散模型
│   ├── egnn_pdbbind_v2016.pt            ✅ (30M) - EGNN 模型
│   └── pk_reg_para.pkl                  ✅ - 参数文件
│
└── logs/
    ├── training_dock_guide_2023_12_17__06_23_35/
    │   └── checkpoints/184000.pt       ✅ (11M) - Binding Affinity Guide
    │
    ├── training_dock_guide_qed_2024_01_06__01_35_21/
    │   └── checkpoints/186000.pt         ✅ (11M) - QED Guide
    │
    └── training_dock_guide_sa_2024_01_20__15_38_49/
        └── checkpoints/162000.pt         ✅ (11M) - SA Guide
```

### 数据文件清单（已就绪）

```
TAGMol/data/
├── test_set -> ../AliDiff/data/test_set     ✅ 测试集 (100个靶点)
│
└── guide -> ../AliDiff/data                 ✅ Guide 训练数据目录
    ├── crossdocked_v1.1_rmsd1.0_pocket10_processed_dock_guide_final.lmdb  ✅ (3.6G)
    ├── crossdocked_pocket10_pose_split_dock_guide.pt                      ✅ (416K)
    └── crossdocked_v1.1_rmsd1.0_pocket10/
        └── index.pkl                                                      ✅ (8K)
```

---

## 🚀 快速开始

### 1. 激活环境

```bash
source /shared/healthinfolab/phz24002/anaconda3/bin/activate
conda activate tagmol
cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH=".":$PYTHONPATH
```

### 2. 验证环境

```bash
# 测试 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 测试 RDKit
python -c "from rdkit import Chem; print('RDKit: OK')"

# 测试 PyTorch Geometric
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

---

## 📊 Evaluation 命令

### 方案 A: 评估 Guide 模型（直接测试 Guide 效果）

```bash
# 评估 Binding Affinity Guide
python scripts/eval_dock_guide.py \
    --ckpt_path logs/training_dock_guide_2023_12_17__06_23_35/checkpoints/184000.pt

# 评估 QED Guide
python scripts/eval_dock_guide.py \
    --ckpt_path logs/training_dock_guide_qed_2024_01_06__01_35_21/checkpoints/186000.pt

# 评估 SA Guide
python scripts/eval_dock_guide.py \
    --ckpt_path logs/training_dock_guide_sa_2024_01_20__15_38_49/checkpoints/162000.pt
```

### 方案 B: 完整流程（采样 + 评估）

#### 步骤 1: 采样（生成分子）

**单个数据点测试（推荐先测试）:**
```bash
# 测试单个数据点 --data_id 0
python scripts/sample_multi_guided_diffusion.py \
    configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml \
    --data_id 0
```

**批量采样所有 100 个测试靶点:**

```bash
# TAGMol 主模型（QED + SA + BA 多目标引导）
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml \
    qed_0.33_sa_0.33_ba_0.34

# Backbone 模型（无 guidance，用于对比）
bash scripts/batch_sample_diffusion.sh \
    configs/sampling.yml \
    backbone
```

**其他配置采样:**
```bash
# 单目标引导 - QED
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_qed_1.yml qed_only

# 单目标引导 - SA
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_sa_1.yml sa_only

# 单目标引导 - BA
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_ba_1.yml ba_only

# 双目标引导 - QED + SA
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_qed_0.5_sa_0.5.yml qed_sa

# 双目标引导 - SA + BA
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_sa_0.5_ba_0.5.yml sa_ba

# 双目标引导 - QED + BA
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_qed_0.5_ba_0.5.yml qed_ba
```

#### 步骤 2: 评估生成的分子

```bash
# 基础评估 (无 docking，最快)
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34 \
    --docking_mode none \
    --protein_root data/test_set

# Vina Score 评估 (推荐，平衡速度和准确性)
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34 \
    --docking_mode vina_score \
    --protein_root data/test_set

# Vina Dock 评估 (最精确但最慢)
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34 \
    --docking_mode vina_dock \
    --protein_root data/test_set \
    --exhaustiveness 16

# QVina 快速评估
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34 \
    --docking_mode qvina \
    --protein_root data/test_set
```

---

## 📋 完整工作流示例

### 快速测试（单个数据点）

```bash
# 1. 激活环境
source /shared/healthinfolab/phz24002/anaconda3/bin/activate
conda activate tagmol
cd /shared/healthinfolab/phz24002/TAGMol
export PYTHONPATH=".":$PYTHONPATH

# 2. 采样单个数据点
python scripts/sample_multi_guided_diffusion.py \
    configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml \
    --data_id 0

# 3. 评估（结果在对应目录的 eval_results 中）
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34_0 \
    --docking_mode vina_score \
    --protein_root data/test_set
```

### 批量评估（所有 100 个测试靶点）

```bash
# 1. 批量采样（运行时间较长，建议在后台运行）
bash scripts/batch_sample_multi_guided_diffusion.sh \
    configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml \
    qed_0.33_sa_0.33_ba_0.34

# 2. 批量评估
python scripts/evaluate_diffusion.py \
    experiments_multi/qed_0.33_sa_0.33_ba_0.34 \
    --docking_mode vina_score \
    --protein_root data/test_set
```

---

## 📁 目录结构

```
TAGMol/
├── configs/                    # 配置文件
│   ├── noise_guide_multi/      # 多目标 guidance 配置
│   │   ├── sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml  # 主模型配置
│   │   ├── sampling_guided_qed_1.yml                      # QED 单目标
│   │   ├── sampling_guided_sa_1.yml                       # SA 单目标
│   │   ├── sampling_guided_ba_1.yml                       # BA 单目标
│   │   └── ...
│   └── sampling.yml            # Backbone 采样配置
│
├── data/                       # 数据目录 (软链接)
│   └── test_set/               # 100个测试靶点
│
├── pretrained_models/          # ✅ 预训练模型已就位
│   ├── pretrained_diffusion.pt
│   └── egnn_pdbbind_v2016.pt
│
├── logs/                       # ✅ Guide 模型已就位
│   ├── training_dock_guide_*/
│   ├── training_dock_guide_qed_*/
│   └── training_dock_guide_sa_*/
│
├── experiments/                # Backbone 采样输出 (自动生成)
├── experiments_multi/          # Guidance 采样输出 (自动生成)
├── scripts/                    # 训练和评估脚本
│   ├── sample_diffusion.py
│   ├── sample_multi_guided_diffusion.py
│   ├── evaluate_diffusion.py
│   ├── eval_dock_guide.py
│   └── batch_*.sh              # 批量运行脚本
│
└── models/                     # 模型定义
```

---

## ⚠️ 注意事项

1. **首次运行 Vina 评估**时会自动准备 `.pdbqt` 和 `.pqr` 文件，耗时较长
2. **数据 ID 范围**: 测试集包含 100 个靶点，ID 范围 0-99
3. **输出目录**:
   - 单个采样: `experiments_multi/[config_name]_[data_id]/`
   - 批量采样: `experiments_multi/[config_name]/`
   - 评估结果: `[sample_path]/eval_results/`
4. **建议**: 先用 `--data_id 0` 测试单个数据点，确保流程正常后再批量运行

---

## 📚 参考

- **论文**: [TAGMol: Target-Aware Gradient-guided Molecule Generation](https://arxiv.org/abs/2406.01650)
- **代码基础**: [TargetDiff](https://github.com/guanjq/targetdiff)
- **资源链接**: [Google Drive](https://drive.google.com/drive/folders/1INaXCjVZCOQ_awNeGl5Xpsde_8JfnJiy)
