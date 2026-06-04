import math

import torch
import torch.nn.functional as F


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9):
    p = torch.clamp((p + eps) / (p.sum(dim=-1, keepdim=True) + eps), min=eps)
    q = torch.clamp((q + eps) / (q.sum(dim=-1, keepdim=True) + eps), min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean() + (q * (q.log() - p.log())).sum(dim=-1).mean()


def router_regularizer(
    probs: torch.Tensor,
    top_k: int = 2,
    entropy_target_factor: float = 0.6,
    entropy_weight: float = 0.2,
    balance_weight: float = 0.05,
    sparsity_weight: float = 0.1,
):
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if probs.dim() != 2:
        raise ValueError("probs must be shape [B, N] or [N]")
    # Be permissive: if caller passes logits-like values, normalize safely.
    if not torch.isfinite(probs).all() or torch.any(probs < 0):
        probs = F.softmax(torch.nan_to_num(probs, nan=0.0), dim=-1)
    row_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    probs = probs / row_sum
    bsz, n_dom = probs.shape
    _ = bsz
    eps = 1e-9
    probs_safe = probs.clamp_min(eps)
    entropy = -(probs_safe * probs_safe.log()).sum(dim=-1).mean()
    h_target = entropy_target_factor * math.log(max(2, n_dom))
    entropy_loss = (entropy - h_target) ** 2

    mean_p = probs.mean(dim=0, keepdim=True)
    uniform = torch.full_like(mean_p, 1.0 / n_dom)
    balance_loss = symmetric_kl(mean_p, uniform)

    k = max(1, min(int(top_k), n_dom))
    topk_mass = probs.topk(k, dim=-1).values.sum(dim=-1).mean()
    sparsity_loss = 1.0 - topk_mass

    total = (
        float(entropy_weight) * entropy_loss
        + float(balance_weight) * balance_loss
        + float(sparsity_weight) * sparsity_loss
    )
    aux = {
        "entropy": float(entropy.detach().item()),
        "H_target": float(h_target),
        "balance_loss": float(balance_loss.detach().item()),
        "sparsity_mass": float(topk_mass.detach().item()),
    }
    if not torch.isfinite(total):
        total = probs.new_tensor(0.0)
    return total, aux

