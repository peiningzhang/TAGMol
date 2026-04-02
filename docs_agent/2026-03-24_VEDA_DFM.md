# task.md: TAGMol Exact Discrete Flow Matching (DFM) Refactoring Task
# 任务书：TAGMol 离散属性 Exact Discrete Flow Matching (DFM) 改造

---

## 1. 任务概述与约束 / Task Overview & Constraints

**[中文]** 本任务旨在将 `TAGMol` 代码库中关于**离散节点特征（如原子类型 `v`）**的生成逻辑，从传统的 DDPM 离散转移矩阵（Transition Matrix）彻底重构为 **Exact Discrete Flow Matching (DFM)**。

**[EN]** This task aims to refactor the generation logic for **discrete node features (e.g., atom types `v`)** in the `TAGMol` codebase, transitioning from traditional DDPM discrete transition matrices to **Exact Discrete Flow Matching (DFM)**.

**核心约束 / Core Constraints:**

1. **彻底清理旧代码 / Thoroughly clean up old code**：废弃所有离散 `transition_matrix`、`log_alpha`、`log_beta` 的马尔可夫链计算逻辑。
   - *Deprecate all discrete Markov chain calculations including `transition_matrix`, `log_alpha`, `log_beta`.*

2. **非线性时间调度 / Non-linear time scheduling**：必须引入时间调度函数 $\kappa_t$ 及其导数 $\dot{\kappa}_t$，绝不能简单使用线性时间 $t$。
   - *Must introduce time scheduling function $\kappa_t$ and its derivative $\dot{\kappa}_t$; never use simple linear time $t$.*

3. **强制预估-校正 / Mandatory Predictor-Corrector**：在推理阶段，必须实现 DFM 论文中基于 $\bar{u}_t = \alpha_t \hat{u}_t - \beta_t \check{u}_t$ 的 Corrector 逻辑（即向前多走一步，再向后退一步注入噪声），此为必选项。
   - *In inference, must implement the DFM paper's Corrector logic based on $\bar{u}_t = \alpha_t \hat{u}_t - \beta_t \check{u}_t$ (step forward, then step back to inject noise). This is mandatory.*

---

## 2. 模块一：实现非线性时间调度器 / Module 1: Non-linear Time Scheduler

**[中文]** **目标**：在 `utils` 或模型初始化中，实现一个离散 Flow Matching 专用的调度器。

**[EN]** **Goal**: Implement a dedicated scheduler for discrete Flow Matching in `utils` or during model initialization.

1. **定义调度函数 / Define scheduling function**：时间 $t \in [0, 1]$。推荐使用类似于 Sigmoid 或余弦调度的平滑曲线。
   - *Time $t \in [0, 1]$. Recommend using smooth curves similar to Sigmoid or cosine scheduling.*

2. **实现以下两个核心方法 / Implement these two core methods**：
   * `kappa(t)`: 返回当前的插值系数 $\kappa_t \in [0, 1]$。($\kappa_0 = 0, \kappa_1 = 1$)
     - *Returns current interpolation coefficient $\kappa_t \in [0, 1]$ where $\kappa_0 = 0, \kappa_1 = 1$*
   * `d_kappa_dt(t)`: 返回 $\kappa_t$ 对时间 $t$ 的精确导数 $\dot{\kappa}_t$。
     - *Returns exact derivative $\dot{\kappa}_t$ of $\kappa_t$ with respect to time $t$*

---

## 3. 模块二：训练前向加噪与 Loss 计算 / Module 2: Training Forward Pass & Loss

**[中文]** **目标**：在 `get_diffusion_loss` 中，根据 DFM 概率插值路径直接生成加噪数据 $v_t$，并计算交叉熵损失。

**[EN]** **Goal**: In `get_diffusion_loss`, generate noised data $v_t$ directly according to the DFM probability interpolation path, and compute cross-entropy loss.

**数学背景 / Mathematical Background**（均匀先验假设 / Uniform prior assumption）：

对于类别数 $S$，干净数据 $v_1$，时间步 $t$，概率路径为：
- *For number of classes $S$, clean data $v_1$, timestep $t$, the probability path is:*

$$p_t(v_t | v_1) = \kappa_t \cdot \text{OneHot}(v_1) + (1 - \kappa_t) \cdot \frac{1}{S}$$

**改造步骤 / Refactoring Steps**：

1. **采样时间 / Sample time**：在批次训练时，为离散属性采样时间步 $t \sim \text{Uniform}(0, 1)$。
   - *During batch training, sample timestep $t \sim \text{Uniform}(0, 1)$ for discrete attributes.*

2. **计算加噪概率 / Compute noising probability**：
   * 计算 `k_t = scheduler.kappa(t)`。
     - *Calculate `k_t = scheduler.kappa(t)`*
   * 构建概率张量 / Build probability tensor：`probs = k_t * one_hot(v_1, S) + (1 - k_t) / S`。

3. **采样 $v_t$ / Sample $v_t$**：使用 `torch.distributions.Categorical(probs).sample()` 得到当前加噪状态 $v_t$。
   - *Use `torch.distributions.Categorical(probs).sample()` to obtain current noised state $v_t$.*

4. **网络预测与损失 / Network prediction and loss**：
   * 网络接收 $(pos_t, v_t, t)$ 等输入，**直接预测干净数据 $v_1$ 的概率分布 (Logits)**。
     - *Network receives $(pos_t, v_t, t)$ and **directly predicts clean data $v_1$ probability distribution (Logits)**.*
   * 损失函数采用标准的交叉熵 / Loss function uses standard cross-entropy：`loss_v = torch.nn.functional.cross_entropy(logits, true_v_1)`。

---

## 4. 模块三：带有预估-校正的推理循环 / Module 3: Predictor-Corrector Inference Loop

**[中文]** **目标**：重构 `sample_diffusion` 中的离散特征采样过程。通过计算精确的边缘概率速率矩阵 (Marginal Rate Matrix) 模拟连续时间马尔可夫链 (CTMC)。

**[EN]** **Goal**: Refactor the discrete feature sampling process in `sample_diffusion`. Simulate Continuous-Time Markov Chain (CTMC) by computing the exact Marginal Rate Matrix.

**超参数定义 / Hyperparameter Definitions**：

* 步长 / Step size $h = \Delta t = 1/N$（例如 $N=100$）。
  - *e.g., $N=100$*
* 噪声注入强度 / Noise injection intensity $\beta$（例如设定为常数 $\beta = 0.1$）。
  - *e.g., constant $\beta = 0.1$*
* 预估步长（前向）/ Predictor step (forward) $h_{fwd} = h \cdot (1 + \beta)$。
* 校正步长（后向）/ Corrector step (backward) $h_{bwd} = h \cdot \beta$。

### 执行循环 / Execution Loop：从 $t = 0$ 积分至 $t = 1$ / Integrate from $t = 0$ to $t = 1$

初始化 / Initialize $v_0 \sim \text{Categorical}(1/S)$。在每个时间步 $t$，执行以下两阶段：
- *At each timestep $t$, execute the following two stages:*

#### 阶段 A: Predictor Step (预估步 / Forward Prediction)

网络预测干净数据的分布 $\hat{p}_\theta(v_1 | v_t)$（经过 Softmax）。
- *Network predicts clean data distribution $\hat{p}_\theta(v_1 | v_t)$ (via Softmax).*

利用公式计算从当前状态 $v_t$ 跳跃到任何其他状态 $y$ ($y \neq v_t$) 的概率：
- *Use formula to compute probability of jumping from current state $v_t$ to any other state $y$ ($y \neq v_t$):*

$$P_{jump}^{fwd}(v_t \to y) = h_{fwd} \cdot \frac{\dot{\kappa}_t \cdot (S-1)}{1 - \kappa_t} \cdot \hat{p}_\theta(v_1=y | v_t)$$

**代码逻辑 / Code Logic**：

1. 计算出所有的 $P_{jump}^{fwd}$（利用 mask 强制 $y = v_t$ 处的概率为 0）。
   - *Compute all $P_{jump}^{fwd}$ (use mask to force probability at $y = v_t$ to 0).*
2. 计算留在原状态的概率 / Compute stay probability：$P_{stay} = 1.0 - \sum_{y \neq v_t} P_{jump}^{fwd}(v_t \to y)$。
   - *(需加入安全截断，若 $P_{stay} < 0$，按比例缩小 Jump 概率使其合法)*
   - *Add safety clamp: if $P_{stay} < 0$, scale down Jump probabilities proportionally to make valid.*
3. 使用 `Categorical` 采样得到前向临时状态 $v'_{t+h}$。
   - *Sample forward temporary state $v'_{t+h}$ using `Categorical`.*

#### 阶段 B: Corrector Step (校正步 / Backward Correction)

**[中文]** 在 DFM 理论中，后向时间采样等于向系统中重新注入向着先验（均匀分布）移动的噪声。执行一次从 $v'_{t+h}$ 退回一步的采样，模拟 Langevin 动态：

**[EN]** In DFM theory, backward time sampling equals re-injecting noise into the system that moves toward the prior (uniform distribution). Perform a one-step backward sampling from $v'_{t+h}$, simulating Langevin dynamics:

**代码逻辑 / Code Logic**：

1. 计算后向跳跃的概率 / Compute backward jump probability。向任意其他状态 $y$ ($y \neq v'_{t+h}$) 的退步跳跃概率可以由均匀先验的后向速度导出，这里给出标准工程实现公式：
   - *Backward jump probability to any other state $y$ ($y \neq v'_{t+h}$) can be derived from uniform prior backward velocity. Standard engineering implementation:*

   $$P_{jump}^{bwd}(v'_{t+h} \to y) = h_{bwd} \cdot \frac{\dot{\kappa}_t}{\kappa_t} \cdot \frac{1}{S}$$

2. 同样地，计算所有的 $P_{jump}^{bwd}$（使用 mask 将 $y = v'_{t+h}$ 的概率置零）。
   - *Similarly, compute all $P_{jump}^{bwd}$ (use mask to set $y = v'_{t+h}$ probability to 0).*

3. 计算停留概率 / Compute stay probability：$P_{stay} = 1.0 - \sum_{y \neq v'_{t+h}} P_{jump}^{bwd}$。

4. 再次使用 `Categorical` 采样，将结果更新为正式的 $v_{t+h}$，并进入下一个时间步循环。
   - *Use `Categorical` to sample again, update result as official $v_{t+h}$, and proceed to next timestep loop.*

---

## 验收清单 / Checklist for Agent

- [ ] 彻底删除了旧版 DDPM 中基于 Transition Matrix 的离散加噪代码。
  - *Thoroughly removed old DDPM discrete noising code based on Transition Matrix.*

- [ ] 实现了非线性 `scheduler` 并能正确求导 $\dot{\kappa}_t$。
  - *Implemented non-linear `scheduler` and can correctly compute derivative $\dot{\kappa}_t$.*

- [ ] 训练时的 Loss 是 `CrossEntropy`，直接以真实干净数据 $v_1$ 作为 target。
  - *Training loss is `CrossEntropy`, directly using real clean data $v_1$ as target.*

- [ ] 采样时严格区分了 Predictor (使用 $h \cdot (1+\beta)$) 和 Corrector (使用 $h \cdot \beta$) 两个动作。
  - *During sampling, strictly distinguished Predictor (using $h \cdot (1+\beta)$) and Corrector (using $h \cdot \beta$).*`

- [ ] 公式中的分母 $1-\kappa_t$ 处添加了 `eps=1e-5` 防止最后一步除以 0。
  - *Added `eps=1e-5` at denominator $1-\kappa_t$ in formula to prevent division by 0 in final step.*
