"""
Muon optimizer - single-device variant.

Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py

Key design decision for TAGMol:
  - We use `SingleDeviceMuonWithAuxAdam` which is a *single optimizer* that
    internally applies Muon to hidden 2-D weight matrices and AdamW to all
    other parameters (embeddings, biases, scalars, output layer).
  - This means the training loop and scheduler code in train_diffusion.py
    require *no changes* — optimizer.step() / optimizer.state_dict() work
    exactly the same as with a plain Adam.

Muon constraints (from the original paper / repo):
  - Requires grad tensors with ndim >= 2.
  - Should NOT be applied to: input embeddings, the final linear head, or
    any bias / scalar parameter.
  - Works best with spectral / unit-norm learning rate; the default lr=0.02
    is in "spectral" units (not the same scale as Adam's lr).
"""

import torch


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (batched quintic iteration)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalisation
    of G.  Runs stably in bfloat16 on GPU.

    For G of shape (..., m, n):
      - Ensures spectral norm ≤ 1 before iterating.
      - Returns something close to U S' V^T where S'_ii ~ Uniform(0.5, 1.5).
    """
    assert G.ndim >= 2, "Muon requires gradient tensors with ndim >= 2"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm ≤ 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Quintic Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute a single Muon parameter update."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:          # conv filters: (out, in, kH, kW) → (out, in*kH*kW)
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Scale so that larger matrices get appropriately sized updates
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


# ---------------------------------------------------------------------------
# Adam helper (used internally for non-Muon parameters)
# ---------------------------------------------------------------------------

def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple,
    eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# ---------------------------------------------------------------------------
# Single-device Muon + aux Adam (unified optimizer, no torch.distributed)
# ---------------------------------------------------------------------------

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Single-device (non-distributed) optimizer combining Muon for hidden weight
    matrices and AdamW for all other parameters.

    Each entry in ``param_groups`` must contain ``use_muon: bool``.

    Muon group keys  : params, lr, momentum, weight_decay, ns_steps, use_muon
    AdamW group keys : params, lr, betas, eps, weight_decay, use_muon

    Example usage (see ``build_muon_optimizer`` helper below for a simpler
    API that automatically splits parameters)::

        hidden_params = [p for n, p in model.named_parameters()
                         if p.ndim >= 2 and 'embed' not in n]
        other_params  = [p for n, p in model.named_parameters()
                         if not (p.ndim >= 2 and 'embed' not in n)]

        muon_group = dict(params=hidden_params, lr=0.02,
                          momentum=0.95, weight_decay=0.0, use_muon=True)
        adam_group = dict(params=other_params,  lr=3e-4,
                          betas=(0.9, 0.95), eps=1e-8,
                          weight_decay=1e-2, use_muon=False)
        optimizer = SingleDeviceMuonWithAuxAdam([muon_group, adam_group])
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group, \
                "Every param_group must contain 'use_muon' (bool)."
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # --- Muon update ---
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    # Flatten conv filters to 2-D before orthogonalisation
                    if g.ndim == 4:
                        g = g.view(g.size(0), -1)
                    if g.ndim < 2:
                        # Scalar / bias: fall back to plain SGD step
                        p.add_(g, alpha=-group["lr"])
                        continue
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    update = muon_update(
                        p.grad.clone(),          # clone so lerp_ doesn't clobber grad
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                    )
                    # Weight decay (decoupled, AdamW-style)
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # --- AdamW update ---
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"]       = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_muon_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adam_lr: float = 3e-4,
    adam_betas: tuple = (0.9, 0.95),
    adam_eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
    muon_exclude_keywords: tuple = ("embed",),
    ns_steps: int = 5,
) -> SingleDeviceMuonWithAuxAdam:
    """
    Automatically partition ``model`` parameters and build a
    ``SingleDeviceMuonWithAuxAdam`` optimizer.

    Parameters eligible for Muon:
      - ndim >= 2  AND
      - none of ``muon_exclude_keywords`` appear in the parameter name.

    All other parameters are optimised with AdamW.

    Args:
        model: The neural network.
        muon_lr: Muon learning rate (spectral units; default 0.02).
        muon_momentum: Muon momentum (default 0.95).
        muon_weight_decay: Muon weight decay (default 0, i.e. no decay).
        adam_lr: AdamW learning rate for non-Muon params.
        adam_betas: AdamW betas.
        adam_eps: AdamW epsilon.
        adam_weight_decay: AdamW weight decay.
        muon_exclude_keywords: Parameter name substrings that force a param
            to be handled by AdamW even if ndim >= 2 (e.g. embeddings, heads).
        ns_steps: Number of Newton-Schulz steps (default 5; more = more exact
            orthogonalisation but slower).

    Returns:
        A ``SingleDeviceMuonWithAuxAdam`` instance.
    """
    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_matrix = param.ndim >= 2
        is_excluded = any(kw in name for kw in muon_exclude_keywords)
        if is_matrix and not is_excluded:
            muon_params.append(param)
        else:
            adam_params.append(param)

    param_groups = [
        dict(
            params=muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            ns_steps=ns_steps,
            use_muon=True,
        ),
        dict(
            params=adam_params,
            lr=adam_lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=adam_weight_decay,
            use_muon=False,
        ),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)
