"""
DDPM vs VEDA 信噪比 (SNR) 对比分析

重要：DDPM 和 VEDA 的 timestep 方向相反！
- DDPM: t=0 (干净) → t=999 (噪声)
- VEDA: t=0 (噪声) → t=999 (干净)

为了在相同"噪声水平"下比较，需要反向对齐 VEDA 的 timestep。
"""

import numpy as np

# DDPM 参数 (sigmoid schedule)
beta_start = 1e-7
beta_end = 2e-3
num_timesteps = 1000

def sigmoid_beta_schedule():
    """DDPM sigmoid beta schedule"""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    betas = np.linspace(-6, 6, num_timesteps)
    betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas

# DDPM 计算
betas = sigmoid_beta_schedule()
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
ddpm_snr = alphas_cumprod / (1.0 - alphas_cumprod)

# VEDA 参数 (EDM)
sigma_min = 0.002
sigma_max = 80.0
sigma_data = 0.5

# VEDA 计算 - log_uniform 映射（注意：方向与 DDPM 相反！）
timesteps = np.linspace(0, num_timesteps-1, num_timesteps)
ratio = timesteps / (num_timesteps - 1)  # 0→1
log_sigma_max = np.log(sigma_max)
log_sigma_min = np.log(sigma_min)
log_sigma = log_sigma_max + ratio * (log_sigma_min - log_sigma_max)
sigma = np.exp(log_sigma)
veda_snr = (sigma_data / sigma) ** 2

# 反向对齐：VEDA t=0 (噪声) 对应 DDPM t=999 (噪声)
# 为了对比相同噪声水平，将 VEDA 反向
veda_snr_aligned = veda_snr[::-1]  # 翻转数组
sigma_aligned = sigma[::-1]

print("=" * 90)
print("DDPM vs VEDA 信噪比对比 (在相同噪声水平下对齐)")
print("=" * 90)
print("注意：VEDA timestep 已反向对齐，使得相同 timestep 代表相同噪声水平")
print()
print(f"{'Val Point':<10} {'Timestep':<10} {'DDPM SNR':<18} {'VEDA SNR':<18} {'对齐方式':<20}")
print("-" * 90)

# 10-step validation
for i in range(10):
    t_ddpm = int(np.linspace(0, num_timesteps-1, 10)[i])
    # VEDA 反向：t_veda = 999 - t_ddpm
    t_veda = num_timesteps - 1 - t_ddpm
    
    ds = ddpm_snr[t_ddpm]
    vs = veda_snr[t_veda]  # 使用反向对齐的 VEDA SNR
    
    marker = "*" if i % 2 == 0 else ""
    align_note = f"DDPM t={t_ddpm} ↔ VEDA t={t_veda}"
    print(f"{i:<10} {t_ddpm:<10} {ds:<18.6f} {vs:<18.6f} {align_note:<20} {marker}")

print("=" * 90)
print("* = 5-step validation 采样点")
print()

# 关键发现
print("=" * 90)
print("关键发现 (相同 timestep = 相同噪声水平)")
print("=" * 90)
print()
print("1. 高噪声区域 (t=0, 刚开始去噪):")
print(f"   DDPM SNR (t=0):      {ddpm_snr[0]:.2f}  (干净)")
print(f"   VEDA SNR (t=999):    {veda_snr[999]:.2f}  (干净) <- 对齐后")
print(f"   DDPM SNR (t=999):    {ddpm_snr[999]:.8f}  (噪声)")
print(f"   VEDA SNR (t=0):      {veda_snr[0]:.8f}  (噪声)")
print()
print("2. 方向对比:")
print("   DDPM: t=0 (α≈1, 干净) → t=999 (α≈0, 噪声)")
print("   VEDA: t=0 (σ=80, 噪声) → t=999 (σ=0.002, 干净)")
print()
print("3. 验证时的问题:")
print("   DDPM validation: timestep=0, 111, 222... 从干净到噪声")
print("   VEDA validation: timestep=0, 111, 222... 从噪声到干净")
print("   → 两者在相同 timestep 数值时处于完全不同的噪声水平！")
print()

# 详细数值分析
print("=" * 90)
print("详细数值对比表 (对齐后)")
print("=" * 90)
print(f"{'Timestep':<12} {'DDPM SNR':<15} {'VEDA SNR':<15} {'VEDA Sigma':<15} {'Ratio':<10}")
print("-" * 90)

for t in [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]:
    ds = ddpm_snr[t]
    # VEDA 反向对齐
    t_veda = num_timesteps - 1 - t
    vs = veda_snr[t_veda]
    sig = sigma[t_veda]
    ratio = ds / vs if vs > 0 else float('inf')
    
    note = "start" if t == 0 else ("end" if t == 999 else "")
    print(f"{t:<12} {ds:<15.6f} {vs:<15.6f} {sig:<15.6f} {ratio:<10.2f} {note}")

print("-" * 90)
print("说明: VEDA Sigma 在 t=0 时为 80 (噪声)，t=999 时为 0.002 (干净)")
print()

# AUROC 影响分析
print("=" * 90)
print("对 AUROC 计算的影响")
print("=" * 90)
print()
print("问题所在:")
print("  在 train_diffusion.py 中，validation 循环是:")
print("    for t in np.linspace(0, model.num_timesteps - 1, 10):")
print()
print("  这意味着:")
print("    DDPM: 从 t=0 (最干净, SNR最高) 开始 → AUROC 容易接近1.0")
print("    VEDA: 从 t=0 (最噪声, SNR最低) 开始 → 初始预测可能随机")
print()
print("解决方案选项:")
print("  选项1: VEDA validation 也使用反向 timestep (999→0)")
print("  选项2: 两者都使用相同的 sigma/SNR 网格，而非 timestep 网格")
print("  选项3: 保持现状，但在分析时注意方向差异")
print()

# 计算反向验证的 AUROC 采样点
print("=" * 90)
print("如果 VEDA 反向验证 (从干净到噪声):")
print("=" * 90)
print(f"{'Step':<8} {'VEDA t':<10} {'VEDA SNR':<15} {'Noise Level':<15}")
print("-" * 90)
veda_val_points = [999, 888, 777, 666, 555, 444, 333, 222, 111, 0]
for i, t in enumerate(veda_val_points):
    snr = veda_snr[t]
    noise_level = "clean" if snr > 1000 else ("medium" if snr > 1 else "noisy")
    print(f"{i:<8} {t:<10} {snr:<15.2f} {noise_level:<15}")
print()
print("这样 VEDA 就和 DDPM 一致：从干净 (高 SNR) 开始测试 AUROC")
print()

# ============================================================================
# Discrete (DFM) 部分分析 - 直接对比 mask_rate / corruption level
# ============================================================================

print("=" * 90)
print("Discrete 部分对比分析 (DFM mask_rate / corruption level)")
print("=" * 90)
print()

# DDPM 离散部分参数
v_beta_s = 0.01  # from configs/training.yml

# DDPM 离散 schedule (cosine)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.sqrt(alphas)

# VEDA DFM: kappa(t) = t / (t + 1), mask_rate = 1 - kappa = 1 / (t + 1)
t_max_dfm = 100.0  # from configs/training.yml
t_dfm = np.linspace(0, t_max_dfm, num_timesteps)
kappa_t = t_dfm / (t_dfm + 1)  # interpolation coefficient
mask_rate_veda = 1 - kappa_t  # corruption/masking rate

# DDPM 离散: 使用 cosine schedule 的 alpha (类似连续部分)
alphas_v = cosine_beta_schedule(num_timesteps, s=v_beta_s)
alphas_cumprod_v = np.cumprod(alphas_v ** 2)  # cumulative product
# DDPM 的 corruption = 1 - alpha_cumprod (roughly)
ddpm_corruption = 1 - alphas_cumprod_v

print("1. DFM (VEDA) 的 mask_rate = 1 - kappa(t) = 1 / (t + 1)")
print("   - t=0:   mask_rate=1.0 (完全噪声/均匀分布)")
print("   - t=100: mask_rate=0.01 (接近真实数据)")
print()

print("2. DDPM 离散部分的 corruption level (1 - alpha_cumprod)")
print("   - t=0:   corruption≈0 (完全真实)")
print("   - t=999: corruption≈1 (接近均匀噪声)")
print()

# 对比表 - 在相同 timestep 下
print("=" * 90)
print("Discrete Corruption/Mask Rate 对比表")
print("=" * 90)
print(f"{'Timestep':<12} {'DDPM Corrupt':<18} {'VEDA MaskRate':<18} {'VEDA kappa':<15} {'方向':<10}")
print("-" * 90)

for t_idx in [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]:
    ddpm_c = ddpm_corruption[t_idx]
    ved_m = mask_rate_veda[t_idx]
    kap = kappa_t[t_idx]
    
    # 方向说明
    if t_idx == 0:
        direction = "DDPM:clean"
    elif t_idx == 999:
        direction = "DDPM:noise"
    else:
        direction = ""
    
    print(f"{t_idx:<12} {ddpm_c:<18.6f} {ved_m:<18.6f} {kap:<15.6f} {direction:<10}")

print("-" * 90)
print()

print("=" * 90)
print("关键发现 (Discrete 部分)")
print("=" * 90)
print()
print("1. DDPM 离散部分:")
print("   - 使用 cosine beta schedule (与连续部分类似)")
print("   - corruption = 1 - alpha_cumprod")
print("   - t=0 时几乎无噪声，t=999 时接近完全噪声")
print()
print("2. VEDA DFM:")
print("   - 使用 kappa(t) = t / (t + 1)")
print("   - mask_rate = 1 / (t + 1)")
print("   - t=0 时完全噪声 (mask_rate=1.0)，t→∞ 时无噪声")
print()
print("3. 方向对比 (与连续部分一致):")
print("   DDPM: t=0 (真实) → t=999 (噪声)")
print("   VEDA: t=0 (噪声) → t=999 (真实)")
print()
print("4. 训练时采样:")
print("   DDPM: timestep ~ Uniform(0, 999)")
print("   VEDA: t ~ Uniform(0, t_max=100)")
print()
print("5. 对验证 AUROC 的影响:")
print("   - 与连续部分相同：VEDA 从 t=0 (噪声) 开始验证会导致初始 AUROC 低")
print("   - 应该在相同 corruption level 下比较 (VEDA 反向对齐)")
print()

# 对齐后的对比
print("=" * 90)
print("对齐后的 Discrete Corruption 对比 (VEDA 反向)")
print("=" * 90)
print(f"{'Timestep':<12} {'DDPM Corrupt':<18} {'VEDA MaskRate':<18} {'差距':<10}")
print("-" * 90)

for t_idx in [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]:
    ddpm_c = ddpm_corruption[t_idx]
    # VEDA 反向对齐：t=0 ↔ t=999
    t_veda_aligned = num_timesteps - 1 - t_idx
    ved_m_aligned = mask_rate_veda[t_veda_aligned]
    gap = abs(ddpm_c - ved_m_aligned)
    
    note = "start" if t_idx == 0 else ("end" if t_idx == 999 else "")
    print(f"{t_idx:<12} {ddpm_c:<18.6f} {ved_m_aligned:<18.6f} {gap:<10.6f} {note}")

print("-" * 90)
print("结论：反向对齐后，两者在相同 timestep 有相似的 corruption level")
print("      这样才可以在验证时公平比较 AUROC")
print()
