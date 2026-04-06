# Diffusion 条件化改造计划

## 目标

为 TAGMol 的 diffusion 训练加入显式 condition，并保留 classifier-free guidance（CFG）路径。

条件来源：

- `vina_score`
- `QED`
- `SA`

计划采用离散分桶后的条件输入，训练时支持有条件 / 无条件两种分支，推理时用标准 CFG 采样。

---

## 设计选择

### 1. 条件表示

- 每个属性单独建一个 embedding：
  - `vina_bin -> vina_embedding`
  - `qed_bin -> qed_embedding`
  - `sa_bin -> sa_embedding`
- 每个属性增加一个 `null` 类，用于无条件分支。
- 三个 embedding 融合成一个 condition token：
  - 推荐 `concat -> MLP -> cond_token`
  - 不建议直接把三个 embedding 直接相加作为唯一表示

### 2. 条件丢弃

- 训练时使用 `p_drop = 0.2`
- 丢弃时将三项 condition 一起置空
- 空条件使用专门的 `null` 类表示，不复用普通桶

### 3. 条件注入方式

- 使用“额外 token 输入”的方式
- 将 `cond_token` prepend 到主序列 / 主图 token 前面
- 后续层通过 attention / message passing 读取该 token

### 4. 推理方式

- 标准 CFG：
  - `pred_cond = model(x, cond)`
  - `pred_uncond = model(x, null)`
  - `pred = pred_uncond + scale * (pred_cond - pred_uncond)`
- 采样时仅改 guidance 分支，不改主扩散流程

---

## 数据处理

### 分桶规则

- 先扫描训练集和测试集，确认样本里是否已经包含这 3 个属性：
  - `vina_score`
  - `QED`
  - `SA`
- 既然当前流程已经支持训练 guide 模型，默认这些属性是存在的，因此这里不需要重新打分，只需要直接读取现有字段。
- 对训练集统计 `vina_score / QED / SA` 的分布
- 采用 5 桶等频切分
- 目标是让每个桶在训练集内样本数尽量接近均匀

### 方向统一

- `QED`：值越大越好
- `SA`：值越大越好
- `vina_score`：值越负越好，需要统一成“越大越优”的语义

### 输出格式

每个样本保存：

- `vina_bin`
- `qed_bin`
- `sa_bin`

训练和验证必须使用同一套桶边界。

---

## 模型改造

### 1. 新增 condition embedding

在模型中加入：

- `vina_embedding`
- `qed_embedding`
- `sa_embedding`
- `null` index 支持

### 2. 新增 condition encoder

将三个 embedding 融合为一个固定维度的 `cond_token`。

建议结构：

```text
vina_emb + qed_emb + sa_emb
        -> concat
        -> MLP
        -> cond_token
```

### 3. 输入接口改造

模型 forward 需要显式接受：

- `vina_bin`
- `qed_bin`
- `sa_bin`
- `cond_drop_flag` 或已处理后的 `cond_token`

无条件分支可通过：

- batch 内统一替换为 `null` 类
- 或由 forward 内部根据 drop mask 处理

---

## 训练流程

### 每个 batch

1. 读取 `vina_bin / qed_bin / sa_bin`
2. 按 `p_drop = 0.2` 随机决定是否置空
3. 构造 `cond_token`
4. 将 `cond_token` 作为额外 token 喂入 diffusion backbone
5. 正常计算 diffusion loss

### 训练目标

- diffusion 主损失不变
- condition 只负责调制生成分布
- 不额外引入分类器损失，保持 CFG 风格

---

## 推理流程

### 采样

1. 输入目标 condition
2. 同时构造一份 `null` condition
3. 对每一步扩散做两次前向：
  - `cond`
  - `uncond`
4. 用 CFG 合成最终预测

### guidance scale

- 默认从 `1.0 ~ 5.0` 扫描
- 先看单属性可控性，再调多属性联合效果

---

## 建议修改文件

优先检查和修改：

- `models/molopt_score_model.py`
- `scripts/train_diffusion.py`
- `scripts/sample_diffusion.py`
- `configs/*.yml`
- 负责数据预处理 / transform 的文件

---

## 验收标准

- 训练时能正确读取三项 condition
- `p_drop = 0.2` 时无条件分支能稳定工作
- `cond_token` 能作为额外 token 进入 backbone
- 推理时可单独跑 cond / uncond
- CFG 采样结果对目标桶有方向性变化

---

## 风险点

- 5 桶可能过粗，条件信息损失较明显
- 桶分布不均衡会影响控制效果
- 三个属性同时控制时可能存在 trade-off，需要后续做权重和 guidance sweep

