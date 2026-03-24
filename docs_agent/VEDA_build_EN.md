# VEDA Implementation Plan: Step-by-Step Migration from TAGMol

> **Objective**: Upgrade TAGMol's dual-track DDPM architecture to the VEDA architecture.
> **Core**: Hybrid dynamics - continuous variables (coordinates pos) use EDM (Variance Exploding Diffusion), discrete variables (node features v) use Discrete Flow Matching.

---

## Execution Guidelines

1. Follow the Phase sequence strictly
2. Report and verify after each completed step
3. Understand the principles before implementation

---

## Phase 1: Configuration and Class Initialization (Foundation)

### Goal

Introduce EDM and Discrete FM parameters in the configuration and differentiate during model initialization.

### Task List

#### 1.1 Modify Configuration Files

**Files**: `configs/training.yml` and `sampling.yml`

Add new fields:
```yaml
model:
  diffusion_type: veda  # or edm_fm

  # EDM coordinate parameters
  sigma_data: 0.5
  sigma_min: 0.002
  sigma_max: 80.0
  rho: 7.0

  # Discrete FM attribute parameters
  discrete_prior: uniform
```

#### 1.2 Modify Model Initialization

**File**: `models/molopt_score_model.py` `__init__` method

```python
if self.diffusion_type == 'veda':
    # Remove original discrete DDPM related calculations
    # - log_alpha
    # - log_beta
    # - transition_matrix

    # Inject Discrete FM prior distribution p_1(v)
    # - marginal prior: frequency of atom types in training set
    # - uniform distribution: uniform distribution
```

---

## Phase 2: EDM Preprocessing and Continuous Coordinate Noising (Continuous Part)

### Goal

Implement the scaling mechanism from the Karras EDM paper for 3D coordinates (pos).

### Task List

#### 2.1 Add New Preprocessing Method

```python
def get_edm_scaling(self, sigma):
    """
    Returns EDM scaling coefficients: c_skip, c_out, c_in, c_noise
    """
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    c_noise = torch.log(sigma) / 4
    return c_skip, c_out, c_in, c_noise
```

#### 2.2 Modify Network Output

Transform the network's output for pos into a denoising prediction:

```
D_θ(x_t, σ) = c_skip · x_t + c_out · F_θ^pos(c_in · x_t, c_noise)
```

---

## Phase 3: Discrete Flow Matching Noising and Path Definition (Discrete Part)

### Goal

Replace the original Markov discrete noising with Flow Matching's probability interpolation path to generate v_t.

### Core Concepts

#### 3.1 Discrete Probability Path

- Time range: t ∈ [0, 1]
  - t = 0: real data
  - t = 1: pure noise
- State v_t class probability:

```
P(v_t | v_0) = (1 - t) · OneHot(v_0) + t · p_1(v)
```

#### 3.2 Implement Noising Function

```python
def sample_discrete_fm_noise(self, v_0, t):
    """
    Perform Categorical sampling according to probability matrix P(v_t | v_0)
    to obtain noised node category v_t
    t can also be written as mask_rate
    """
    prob = (1 - t) * F.one_hot(v_0, num_classes) + t * self.prior_dist
    return torch.distributions.Categorical(prob).sample()
```

---

## Phase 4: Joint Training Process and Dual Loss Computation (Training)

### Goal

Modify `get_diffusion_loss` to integrate EDM Loss and Discrete FM Loss.

### Task List

#### 4.1 Time Step and Sigma Alignment

```python
# Continuous coordinate sampling
sigma = LogNormal(P_mean, P_std).sample()

# Discrete attribute sampling
t = Uniform(0, 1).sample()

# Mapping method (optional)
t = f(sigma) = sigma/(sigma + 1)
```

#### 4.2 Coordinate EDM Loss

```python
error = D_theta(pos_t, sigma) - pos_0
loss_pos = weight(sigma) * MSE(error)
```

#### 4.3 Node Attribute Discrete FM Loss

```python
# Network receives (pos_t, v_t, t), outputs Logits
predicted_logits = network(pos_t, v_t, t)

# Target is to predict real data v_0
loss_v = CrossEntropy(predicted_logits, true_v_0)
```

#### 4.4 Overall Loss

```
Loss = loss_pos + lambda_v · loss_v
```

---

## Phase 5: Joint Sampler and Inference Process Refactoring (Joint Sampling)

### Goal

Refactor `sample_diffusion` to simultaneously use:

- **Heun stepper**: solve pos SDE/ODE
- **Discrete Euler stepper**: solve v probability flow

### Sampling Flow

#### 5.1 Initialization

```python
pos_T ~ N(0, sigma_max^2)
v_T ~ Prior(p_1)  # Sample from marginal or uniform distribution
```

#### 5.2 Time Step Scheduling Strategy

**Naming Convention**:
- Current time step distribution named `log_uniform` (default)
- Preserve interface to support other schedulers (such as `arcsin`, `edm`, `edm1`, etc.)

```python
# Configurable scheduler
time_scheduler: str = "log_uniform"  # Options: ["log_uniform", "arcsin", "edm", "edm1"]
```

#### 5.3 Time Step Descending Loop (t: 1 → 0, corresponding to σ_max → σ_min)

**For continuous coordinates pos (Heun/Euler):**

```python
d_i = (pos_i - D_theta(pos_i, sigma_i)) / sigma_i
pos_next = pos_i + step_size * d_i  # First-order step
```

**For discrete attributes v (Discrete Probability ODE Solver):**

```python
# Network outputs predicted probability for v_0: p_hat_0
p_hat_0 = softmax(network(pos_i, v_i, t))

# Compute reverse transition probability
P(v_{t-Δt} | v_t) = Σ_{v_0} P(v_{t-Δt} | v_0) · p_hat(v_0 | v_t) / P(v_t | v_0)

# Sample v_next from transition probability
v_next = sample_from(P(v_{t-Δt} | v_t))
```

#### 5.3 Iterative Update

Use the newly generated `pos_next` and `v_next` as inputs for the next step.

---

## Phase 6: Redundant Code Cleanup and Guidance Generation (Cleanup & Guidance)

### Goal

Clean up old code and adapt conditional guidance.

### Task List

#### 6.1 Remove Old Code

Delete unused discrete functions:

- `q_v_posterior`
- `_predict_v0_from_posterior`
- Other DDPM-related helper functions

#### 6.2 Attribute Guidance

**EDM (coordinates) guidance:**

```python
d_i = d_i + guidance_scale · ∇_pos E
```

**Discrete FM (attributes) guidance:**

When computing the transition probability for `v_next`, inject classifier gradients (or energy gradients) from the attribute model to adjust the logits distribution of `p_hat(v_0 | v_t)`.

---

## Appendix: How to Send to Agent

> "This is the complete refactoring plan. It includes EDM (Variance Exploding) for continuous coordinates and Discrete Flow Matching (Discrete FM) for discrete node attributes. The combination of these two is the complete VEDA architecture. Please read carefully and provide specific code-level modification plans starting from Phase 1."
