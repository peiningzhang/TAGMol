# TAGMol 改造为 VEDA 的分步实施计划

> **目标**：将 TAGMol 的双轨 DDPM 架构升级为 VEDA 架构。
> **核心**：混合动力学 - 连续变量（坐标 pos）使用 EDM（方差爆炸扩散），离散变量（节点特征 v）使用 Discrete Flow Matching（离散流匹配）。

---

## 执行须知

1. 严格按照 Phase 顺序执行
2. 每完成一步请汇报并验证
3. 先理解原理再动手实现

---

## Phase 1: 配置文件与类初始化适配（Foundation）

### 目标

在配置中引入 EDM 和 Discrete FM 的参数，并在模型初始化时进行区分。

### 任务清单

#### 1.1 修改配置文件

**文件**: `configs/training.yml` 和 `sampling.yml`

新增字段：

```yaml
model:
  diffusion_type: veda  # 或 edm_fm

  # EDM 坐标参数
  sigma_data: 0.5
  sigma_min: 0.002
  sigma_max: 80.0
  rho: 7.0

  # Discrete FM 属性参数
  discrete_prior: uniform
```

#### 1.2 修改模型初始化

**文件**: `models/molopt_score_model.py` 的 `__init__` 方法

```python
if self.diffusion_type == 'veda':
    # 移除原有离散 DDPM 相关计算
    # - log_alpha
    # - log_beta
    # - transition_matrix

    # 注入 Discrete FM 的先验分布 p_1(v)
    # - marginal prior: 训练集各类原子的频率
    # - uniform distribution: 均匀分布
```

---

## Phase 2: EDM 预处理与连续坐标的加噪（Continuous Part）

### 目标

为三维坐标 (pos) 实现 Karras EDM 论文中的缩放机制。

### 任务清单

#### 2.1 新增预处理方法

```python
def get_edm_scaling(self, sigma):
    """
    返回 EDM 缩放系数: c_skip, c_out, c_in, c_noise
    """
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    c_noise = torch.log(sigma) / 4
    return c_skip, c_out, c_in, c_noise
```

#### 2.2 修改网络输出

将网络对 pos 的输出转换为降噪预测值：

```
D_θ(x_t, σ) = c_skip · x_t + c_out · F_θ^pos(c_in · x_t, c_noise)
```

---

## Phase 3: Discrete Flow Matching 的加噪与路径定义（Discrete Part）

### 目标

取代原有的马尔可夫离散加噪，使用 Flow Matching 的概率插值路径生成 v_t。

### 核心概念

#### 3.1 离散概率路径

- 时间范围：t ∈ [0, 1]
  - t = 0: 真实数据
  - t = 1: 纯噪声
- 状态 v_t 的类别概率：

```
P(v_t | v_0) = (1 - t) · OneHot(v_0) + t · p_1(v)
```

#### 3.2 实现加噪函数

```python
def sample_discrete_fm_noise(self, v_0, t):
    """
    根据概率矩阵 P(v_t | v_0) 进行 Categorical 采样
    得到加噪后的节点类别 v_t
    t也可以写作mask_rate
    """
    prob = (1 - t) * F.one_hot(v_0, num_classes) + t * self.prior_dist
    return torch.distributions.Categorical(prob).sample()
```

---

## Phase 4: 联合训练过程与双重 Loss 计算（Training）

### 目标

修改 `get_diffusion_loss`，整合 EDM Loss 和 Discrete FM Loss。

### 任务清单

#### 4.1 时间步与 Sigma 的对齐

```python
# 连续坐标采样
sigma = LogNormal(P_mean, P_std).sample()

# 离散属性采样
t = Uniform(0, 1).sample()

# 映射方式（可选）
t = f(sigma) = sigma/(sigma + 1)  
```

#### 4.2 位置坐标的 EDM Loss

```python
error = D_theta(pos_t, sigma) - pos_0
loss_pos = weight(sigma) * MSE(error)
```

#### 4.3 节点属性的 Discrete FM Loss

```python
# 网络接收 (pos_t, v_t, t)，输出 Logits
predicted_logits = network(pos_t, v_t, t)

# 目标是预测真实数据 v_0
loss_v = CrossEntropy(predicted_logits, true_v_0)
```

#### 4.4 整体损失

```
Loss = loss_pos + lambda_v · loss_v
```

---

## Phase 5: 联合采样器与推理过程重构（Joint Sampling）

### 目标

重构 `sample_diffusion`，同时使用：

- **Heun 步进**：解 pos 的 SDE/ODE
- **Discrete Euler 步进**：解 v 的概率流

### 采样流程

#### 5.1 初始化

```python
pos_T ~ N(0, sigma_max^2)
v_T ~ Prior(p_1)  # 根据边缘分布或均匀分布采样
```

#### 5.2 时间步调度策略

**命名规范**：
- 当前时间步分布命名为 `log_uniform`（默认）
- 保留接口以支持其他 scheduler（如 `arcsin`, `edm`, `edm1` 等）

```python
# 可配置的 scheduler
time_scheduler: str = "log_uniform"  # 可选: ["log_uniform", "arcsin", "edm", "edm1"]
```

#### 5.3 时间步递减循环（t: 1 → 0，对应 σ_max → σ_min）

**对于连续坐标 pos (Heun/Euler)：**

```python
d_i = (pos_i - D_theta(pos_i, sigma_i)) / sigma_i
pos_next = pos_i + step_size * d_i  # 一阶步进
```

**对于离散属性 v (Discrete Probability ODE Solver)：**

```python
# 网络输出对 v_0 的预测概率 p_hat_0
p_hat_0 = softmax(network(pos_i, v_i, t))

# 计算逆向转移概率
P(v_{t-Δt} | v_t) = Σ_{v_0} P(v_{t-Δt} | v_0) · p_hat(v_0 | v_t) / P(v_t | v_0)

# 根据转移概率采样 v_next
v_next = sample_from(P(v_{t-Δt} | v_t))
```

#### 5.3 迭代更新

将新生成的 `pos_next` 和 `v_next` 作为下一步的输入。

---

## Phase 6: 冗余代码清理与引导生成（Cleanup & Guidance）

### 目标

清理旧代码，适配条件引导。

### 任务清单

#### 6.1 移除旧代码

删除不再使用的离散函数：

- `q_v_posterior`
- `_predict_v0_from_posterior`
- 其他 DDPM 相关辅助函数

#### 6.2 属性引导（Guidance）

**EDM (坐标) 的引导：**

```python
d_i = d_i + guidance_scale · ∇_pos E
```

**Discrete FM (属性) 的引导：**

在计算 `v_next` 的转移概率时，注入属性模型的分类器梯度（或能量梯度），调整 `p_hat(v_0 | v_t)` 的 logits 分布。

---

## 附录：如何发送给 Agent

> "这是修改后的完整重构计划。它包含了连续坐标的 EDM (Variance Exploding) 和离散节点属性的 Discrete Flow Matching (Discrete FM)，这两者结合才是完整的 VEDA 架构。请你仔细阅读，并从 Phase 1 开始提供代码层面的具体修改方案。"

