# TAGMol 架构改造：显式引入化学键生成方案 (Explicit Bond Generation)

当前 TAGMol (及各种类似 DiffDock、EDM 的纯坐标扩散模型) 主要依赖**隐式成键 (Implicit Bonds)**：即只预测离散的原子种类 (Categorical) 和由连续变量表示的 3D 坐标位置，最后通过启发式的距离查表 (`check_stability` 函数等) 推出化学键。

如果希望在架构中显式地进行化学键（Edge/Bond）的扩散与生成，以下是基于原版扩散模型架构设计的具体落地方案与架构复杂度分析。

## 结论一瞥
**总体而言，修改底层架构来支持显式化学键生成的工作量可控（约更改20%逻辑），且并不需要对 GNN 内部的各隐层进行伤筋动骨的修改。** 
由于化学键预测的范围仅局限于药物配体原子（$N_{lig} \le 50$），因此增加预测头的 $O(N_{lig}^2)$ 计算开销极其微小。

## 架构层面的四大改动核心

### 1. 网络前端 (Input & Initial Layer) 
在进入第一层 GNN 更新前，当前时间步含噪的边分类状态张量 $E_t$ 必须注入模型：
*   **方法**：使用 Embedding 层将离散的初始边特征投射后，直接把边信息拼接到配对消息模块 (Pairwise Message Module) 内，使其融合或“吸收”进节点初始隐藏特征 $h_0$ 中。
*   **好处**：这样做免除了后续数百维的特征要在整张图的层层流转中维持 $N_{lig} \times N_{lig}$ 的全边特征开销，同时让第一层网络感知到边此时的噪声情况。

### 2. 深层 GNN 不变 (Intermediate Message Passing)
*   **方法**：模型内部那 L 层的 `ScorePosNet3D`、`EGNN` 或 `Transformer` 等核心组块**保持完全不变**。深层交互依然是轻度且高效的基于截断半径的节点消息传递操作（仅更新 Node）。不再做专门的一维 Edge 更新传递。

### 3. 网络输出头 (Edge Predictor Head / Refinement)
*   **方法**：在深层特征完成所有传播并输出最终的节点高阶特征 $h_i, h_j$ 后，在系统末端外挂一个类似于 Multi-Layer Perceptron (MLP) 构成的新“化学键预测头”。
*   **逻辑**：把原子 $i$、$j$ 的特征与网络最终推断出来的物理向量距离 $d_{ij} = \|x_i - x_j\|$ 相结合： `MLP([h_i, h_j, d_{ij}]) -> R^C`。
*   **结果**：MLP 输出 $N_{lig} \times N_{lig} \times C$ 大小的 Logits 矩阵以表征单键、双键等各类化学键的概率。由于这里的配体原子数量小，矩阵运算几乎不消耗性能；且由于基于特征 $h$ 与标量 $d$，预测天然符合 SE(3)/E(3) 的等变性要求。

### 4. 损失 (Loss) 与 采样 (Sampler)
*   必须恢复在 `models/molopt_score_model.py` 中被注释掉的 `bond_loss` 系列计算。对生成的边执行针对 Ground Truth 的类别交叉熵（Cross-Entropy）惩罚。需要注意由于绝大多原子对没有键，需要仔细设置“No bond”的权重比例惩罚平衡（`loss_non_bond_weight`）。
*   修改 `sample_diffusion.py`，使推理时不仅更新位置 $x$ 与类型 $v$，同样要有一步按照离散马尔可夫转移矩阵（例如 Mask 生成、全类别 Uniform）从 $E_t$ 对化学键类型进行去噪跳变计算出 $E_{t-1}$。

---

## 数据预处理与 GT Bond 提取

### 5.1 现有 Bond 数据的完整流向（现状）

Ground Truth 化学键信息在原始代码中**已经全程保存，但目前仅作为重建后处理的参照，并未被送入模型训练**。完整链路如下：

| 阶段 | 文件 | 关键代码 | 说明 |
|------|------|----------|------|
| **原始解析** | `utils/data.py` | `parse_sdf_file()` | 从 `.sdf`/`.mol2` 读取配体，用 RDKit 提取所有化学键，存为双向 COO 格式 |
| **Bond 类型映射** | `utils/data.py` | `BOND_TYPES` 字典 | `{UNSPECIFIED:0, SINGLE:1, DOUBLE:2, TRIPLE:3, AROMATIC:4}`，共 5 类 |
| **存入 Data 对象** | `utils/data.py` | 返回字段 `bond_index`, `bond_type` | 后续在 `pl_pair_dataset.py` 中分别以 `ligand_bond_index` / `ligand_bond_type` 挂载到 `ProteinLigandData` |
| **Dataset 处理** | `datasets/pl_pair_dataset.py` | `_process()` | 调用 `parse_sdf_file` → `torchify_dict` → 写入 LMDB，`bond_index`/`bond_type` 已在其中 |
| **One-hot 编码** | `utils/transforms.py` | `FeaturizeLigandBond.__call__()` | `F.one_hot(data.ligand_bond_type - 1, num_classes=5)` → `data.ligand_bond_feature` |
| **DataLoader batch 增量** | `datasets/pl_data.py` | `__inc__` 中 `ligand_bond_index` | 保证 batch 拼接后 bond index 正确偏移 |
| **连接矩阵工具** | `datasets/pl_data.py` | `get_batch_connectivity_matrix()` | 将稀疏 COO 格式展开为 `N×N` 稠密整数矩阵，0=无键，1-4=各键型 |

> **结论**：所有 GT bond 数据已经贯穿整个数据管线，改动的目标是**激活**这条链路并把它接入训练目标。

---

### 5.2 当前未激活的部分

以下两处是目前被「注释掉」或「未接通」的关键节点：

**① `FOLLOW_BATCH` 缺失 `ligand_bond_type`**
在 `datasets/pl_data.py` 第 7 行：
```python
# 现状（不完整）
FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)
```
`ligand_bond_type` 已在其中，但 **`ligand_bond_index` 本身没有被 follow**——这意味着 batch 化后无法直接用 `ligand_bond_batch` 来切分 bond 所属的分子。需确认 `DataLoader` 的 `follow_batch` 参数是否被正确传递（见 `ProteinLigandDataLoader`）。

**② `FeaturizeLigandBond` 默认不在 transform 链中**
在训练脚本的 `transform` 构建处（`scripts/train_diffusion.py` / `train_consistency.py`），目前的 transform 列表通常只包含：
```python
transforms.FeaturizeProteinAtom(),
transforms.FeaturizeLigandAtom(mode=config.data.transform.ligand_atom_mode),
transforms.RandomRotation(),
```
`FeaturizeLigandBond` 从未被加入，因此 `data.ligand_bond_feature`（one-hot 格式的 GT bond）在训练时始终为空。

**③ `molopt_score_model.py` 中 bond loss 相关代码被注释**
第 228-231 行和第 236-237 行的 `bond_loss`、`bond_net_type`、`loss_bond_weight` 等字段均已注释，需要解注释并接入新的 bond 预测头输出。

---

### 5.3 需要新增的处理：稠密目标张量构建

模型末端的 Bond Predictor Head 输出的是 $N_{lig} \times N_{lig} \times C$ 的 Logits 矩阵（对称），损失需要与之对齐的稠密 GT 矩阵。现有的 `get_batch_connectivity_matrix()` 已实现这一转换，但仅用于评估，需将其纳入训练 forward 流程：

```python
# 在 get_diffusion_loss() 中，于 preds 计算后插入：
from datasets.pl_data import get_batch_connectivity_matrix

# ligand_bond_batch 由 FOLLOW_BATCH 中的 'ligand_bond_type' 生成
bond_gt_matrices = get_batch_connectivity_matrix(
    ligand_batch,
    ligand_bond_index,      # shape: (2, num_bonds_in_batch)
    ligand_bond_type,       # shape: (num_bonds_in_batch,)
    ligand_bond_type_batch  # shape: (num_bonds_in_batch,)  —— DataLoader 自动生成
)
# bond_gt_matrices: list of (N_i x N_i) LongTensor，值域 {0,1,2,3,4}
```

随后将其 stack 为统一格式后计算加权 CrossEntropy（注意「无键」类别 0 的样本占绝大多数，需设置 `weight` 参数压低其梯度权重）：

```python
# 示例：计算 bond loss（pred_bond_logits 由 Bond Head 输出，shape: (N_lig, N_lig, C)）
loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.1)
bond_weight = torch.ones(5, device=device)        # C=5 类
bond_weight[0] = loss_non_bond_weight             # 降低「无键」类别权重
loss_bond = F.cross_entropy(
    pred_bond_logits.view(-1, 5),                 # (N_lig*N_lig, C)
    bond_gt.view(-1),                             # (N_lig*N_lig,)
    weight=bond_weight
)
```

---

### 5.4 改造步骤清单（✅ 已全部实施完成）

- [x] **Step 1**：`FeaturizeLigandBond()` 已存在于 `scripts/train_diffusion.py` transform 链（第 167 行）。`data.ligand_bond_feature` shape 验证：`torch.Size([130, 5])` ✓

- [x] **Step 2**：`FOLLOW_BATCH` 已包含 `'ligand_bond_type'`；`batch.ligand_bond_type_batch` shape=`torch.Size([130])` = 2 分子总 bond 数 ✓

- [x] **Step 3** — `models/molopt_score_model.py` 实际改动：
  - 解注释并启用 `bond_loss`、`loss_bond_weight`、`loss_non_bond_weight` 超参（`getattr` 安全读取）
  - 新建 Bond Head MLP：`Linear(257→128) → ShiftedSoftplus → Linear(128→64) → ShiftedSoftplus → Linear(64→5)`（输入维 = `hidden_dim*2+1 = 257`）
  - `forward()` 末尾新增：按图循环计算 `(N_g, N_g, 5)` bond logits，存入 `preds['pred_bond_logits']`

- [x] **Step 4** — `get_diffusion_loss()` 实际改动：
  - 签名加入 `ligand_bond_index`, `ligand_bond_type`, `ligand_bond_type_batch` 参数
  - 调用 `get_batch_connectivity_matrix()` 构建稠密 `(N_g, N_g)` GT 矩阵（值域 `{0..4}`）
  - 按图循环计算加权 CrossEntropy（`no_bond_weight=0.1`），纳入总 loss
  - 返回字典新增 `'loss_bond'` 字段

- [x] **Step 5** — `configs/training.yml` 改动：
  - 迁移至 `model:` 节：`bond_loss: true`, `loss_bond_weight: 1.0`, `loss_non_bond_weight: 0.1`
  - 移除 `train:` 节中原有的冗余 `bond_loss_weight`

- [x] `scripts/train_diffusion.py` 同步改动：`train()` / `validate()` 传入 bond 参数，日志新增 `bond` 字段

---

### 5.5 验证结果（Smoke Test）

```
Batch: protein_atoms=802, ligand_atoms=62
ligand_bond_index shape: torch.Size([2, 130])
ligand_bond_type_batch shape: torch.Size([130])

loss_pos : 2.780252
loss_v   : 2.551770
loss_bond: 1.609789   ← 非零、非NaN ✓
loss     : 10.716078

✅ Smoke test PASSED
```
