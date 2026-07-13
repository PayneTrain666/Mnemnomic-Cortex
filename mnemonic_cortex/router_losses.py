"""
Plain-language summary
----------------------
What this file is for: Training losses that keep routers well-behaved.
How it fits in the system: Regularizes routing decisions during learning.
Status: ACTIVE when advanced routing trains
Important notes for non-coders: Not a runtime memory store.
"""

import math

import torch


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9):
    p = (p + eps) / (p.sum(dim=-1, keepdim=True) + eps)
    q = (q + eps) / (q.sum(dim=-1, keepdim=True) + eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean() + (q * (q.log() - p.log())).sum(dim=-1).mean()


def router_regularizer(
    probs: torch.Tensor,
    top_k: int = 2,
    entropy_target_factor: float = 0.6,
    entropy_weight: float = 0.2,
    balance_weight: float = 0.05,
    sparsity_weight: float = 0.1,
):
    bsz, n_dom = probs.shape
    _ = bsz
    eps = 1e-9
    entropy = -(probs * (probs.add(eps).log())).sum(dim=-1).mean()
    h_target = entropy_target_factor * math.log(max(2, n_dom))
    entropy_loss = (entropy - h_target) ** 2

    mean_p = probs.mean(dim=0, keepdim=True)
    uniform = torch.full_like(mean_p, 1.0 / n_dom)
    balance_loss = symmetric_kl(mean_p, uniform)

    k = min(int(top_k), n_dom)
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
    return total, aux

