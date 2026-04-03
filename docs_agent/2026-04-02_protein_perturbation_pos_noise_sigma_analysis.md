# 蛋白 pocket 扰动 `pos_noise_std = 0.1`：是否可与时间 \(t\)（或 σ）挂钩？

本文档说明训练中对蛋白坐标施加的固定高斯扰动（配置项 `train.pos_noise_std`，默认 **0.1**）的语义，并讨论将其与扩散噪声级别 **σ**（或与 σ 单调对应的“有效时间”）联动的几种方案。**仅为设计分析**，不涉及具体代码改动。

## 1. 当前行为（简要）

实现位置概览：

- 训练：`scripts/train_diffusion.py`（及 `scripts/train_dock_guide.py` 等）在调用 `get_diffusion_loss` / `get_loss` 前，对 `batch.protein_pos` 加上 `torch.randn_like * config.train.pos_noise_std`。
- 配体：在 `models/molopt_score_model.py` 的 `get_diffusion_loss`（及 guide 模型中）按 **VEDA/EDM** 的 **σ** 对配体位置加噪。
- 配置：`configs/training_cfg.yml` 等中 `train.pos_noise_std: 0.1`。
- 采样：`scripts/sample_diffusion.py` 等通常将数据里的 **`batch.protein_pos` 原样**作为条件，**不再**加该训练用扰动。

| 对象 | 训练时噪声 | 与扩散时间的关系 |
|------|------------|------------------|
| **配体位置** | \(\text{ligand} \leftarrow \text{ligand} + \sigma \varepsilon\)，σ ~ LogNormal 等 | **强相关**：σ 即 EDM 的噪声级别 |
| **蛋白位置** | \(\text{protein} \leftarrow \text{protein} + 0.1 \cdot \varepsilon'\) | **无关**：固定标准差，与当前 σ / timestep 独立 |

因此存在一点 **train / infer 条件分布差异**：训练时 pocket 带独立高斯噪声，推理时常为干净 pocket。可视为一种 **条件增强**，但其强度与当前 **σ** 未对齐。

## 2. 和 \(t\)（或 σ）挂钩的动机

1. **尺度一致**：σ 从很小到很大，配体“看不清结构”的程度在变，而 pocket 扰动若一直固定为 0.1（在同一坐标系下），可能与当前扩散阶段在尺度上不协调。
2. **课程式难度**：高噪声时可略增大 pocket 扰动以提高鲁棒性；低噪声时减小扰动，更贴近精细几何与 **无噪推理** 使用的 pocket。
3. **减轻 train–infer 对齐压力**：若推理用干净 pocket，训练后期若 pocket 也更接近干净，有利于行为一致。

## 3. 可选改动方案（概念层面）

以下均指**训练阶段**对传入网络的蛋白坐标扰动；实现时应对 **同一复合物（图）** 使用 **与同一步配体加噪相同的 σ**。

### 方案 A：与 σ 成比例（推荐优先尝试）

\[
\sigma_{\text{pocket}} = \lambda \cdot \sigma
\]

- **含义**：配体越噪，条件 pocket 也越“糊”，促使模型在高 σ 下不过分依赖绝对原子坐标。
- **优点**：与 EDM 变量自然同量纲；主要超参为 \(\lambda\)。
- **风险**：σ 很大时 pocket 位移过大可能带来不合理几何（互穿等），通常需要 **clamp** 或对 \(\lambda\) 仔细调。
- **与采样的关系**：推理时 σ 沿Schedule变小，若采用同一公式，pocket 扰动可自然减弱（若仅在训练内部对条件加噪，则推理仍常为干净 pocket，需在训练中显式设计“后期小扰动”以贴近推理）。

### 方案 B：σ 的凸组合（在固定 0.1 附近微调）

\[
\sigma_{\text{pocket}} = \text{pos\_noise\_std} \cdot \bigl( \alpha + (1-\alpha)\, g(\sigma) \bigr)
\]

其中 \(g(\sigma)\) 可归一化到 \([0,1]\)（例如用 \(\log\sigma\) 相对 \(\log\sigma_{\max},\log\sigma_{\min}\)）。

- **含义**：保留现有 0.1 量级作为基准，再让一部分随 σ 变化。
- **优点**：改动温和，便于做 ablation。
- **风险**：超参略多。

### 方案 C：只依赖离散时间索引 \(t\)

\[
\sigma_{\text{pocket}} = h\!\left(\frac{t}{T}\right) \cdot \sigma_0
\]

- **注意**：VEDA 训练里若 σ 为连续随机抽样，未必与某个离散 \(t\) 一一对应；若强行用名义上的 \(t\) 而不与 σ 对齐，容易出现“时间标签”与真实噪声级别不一致。一般 **更推荐直接用 σ** 定义调度，而非单独虚构 \(t\)。

### 方案 D：分段 / 阈值调度

- 例如：高 σ 区间用较大（或维持 0.1）的 pocket 扰动；低 σ 区间用 0 或很小的扰动。
- **优点**：后期可接近 **clean pocket**，贴近推理条件。
- **风险**：若在 σ 边界上跳变过猛，可能对优化稳定性不利。

### 方案 E：与 `center_pos_rescale`、`rescale_factor` 一致

模型内会对蛋白与配体做 `center_pos_rescale`。扰动是在**调用模型前**加在 raw batch 上，还是应在 rescale 后的空间定义，会改变“0.1”的实际物理含义。若将 \(\sigma_{\text{pocket}}\) 与 σ 挂钩，应明确 **与配体 EDM 在同一坐标语义下** 解释，避免混用空间。

## 4. 与扩散理论是否冲突

- Pocket **不是**当前公式里对条件变量显式写出的 forward SDE 状态；更接近 **数据增广**。将 \(\sigma_{\text{pocket}}\) 设为 \(\sigma\) 的函数 **不会**直接破坏配体 EDM 的推导，但会改变条件分布 \(p(\text{ligand}\mid\text{pocket},\sigma)\)。
- 实现上须保证：**同一次前向中**，蛋白扰动所使用的噪声级别与 **该 batch 当前用于配体的 σ** 一致，避免条件–目标错位。

## 5. 实验与验证建议

1. **Ablation**：例如方案 A（\(\lambda\sigma\) + clamp）对比 baseline 固定 0.1，看验证集 loss 与采样指标。
2. **推理对齐**：明确采样是否始终用干净 pocket；若希望一致，可考虑训练后期 \(\sigma_{\text{pocket}}\to 0\) 或小 \(\lambda\)。
3. **多脚本一致性**：若同时训练 score 与 dock guide，两者若都使用 `pos_noise_std`，新调度应对 **两条管线** 一并评估，避免只改其一导致分布错配。

## 6. 小结

- **可以**将蛋白扰动与 **σ**（或与 σ 单调对应的有效扩散阶段）挂钩；从实现与直觉上 **方案 A（\(\lambda\sigma\) + 必要时 clamp）** 最直接。
- **固定 0.1** 的优点是简单，作为与 σ 无关的随机口袋抖动；缺点是与扩散阶段尺度 **解耦**，且与 **无噪推理 pocket** 的差异无法随 σ 解释。
- 是否值得改，取决于当前瓶颈是否来自“条件过稳 / 过乱”或 train–infer 条件偏移；若高 σ 过拟合 pocket 细节或低 σ 对几何过敏，σ 联动扰动值得尝试。
