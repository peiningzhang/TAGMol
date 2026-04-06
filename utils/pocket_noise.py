"""Training-time Gaussian noise on protein (pocket) coordinates in raw coordinate space."""

import torch


def apply_training_pocket_pos_noise(
    protein_pos,
    batch_protein,
    sigma,
    *,
    mode,
    pos_noise_std,
    sigma_coeff,
    max_pocket_noise,
):
    """
    Args:
        protein_pos: (N_prot, 3)
        batch_protein: (N_prot,) graph index per protein atom
        sigma: (num_graphs,) EDM noise level per graph (same as ligand sigma)
        mode: 'fixed' | 'sigma_scaled' | 'sigma_sqrt_scaled'
        pos_noise_std: scalar std when mode == 'fixed'
        sigma_coeff: multiplier for sigma-scaled modes
        max_pocket_noise: clamp upper bound for sigma-scaled modes (A-scale in data coords)
    """
    device = protein_pos.device
    dtype = protein_pos.dtype
    if mode == 'fixed':
        noise_scale = torch.full(
            (protein_pos.shape[0], 1), float(pos_noise_std), device=device, dtype=dtype
        )
    elif mode == 'sigma_scaled':
        std_g = torch.clamp(
            sigma * float(sigma_coeff), min=0.0, max=float(max_pocket_noise)
        )
        noise_scale = std_g[batch_protein].unsqueeze(-1)
    elif mode == 'sigma_sqrt_scaled':
        std_g = torch.clamp(
            torch.sqrt(sigma.clamp(min=0.0)) * float(sigma_coeff),
            min=0.0,
            max=float(max_pocket_noise),
        )
        noise_scale = std_g[batch_protein].unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown pocket_noise_mode: {mode!r} (expected 'fixed', 'sigma_scaled', or 'sigma_sqrt_scaled')"
        )

    noise = torch.randn_like(protein_pos) * noise_scale
    return protein_pos + noise
