import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unit_complex_real(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize real-packed complex vectors.
    z: [..., 2q], interpreted as complex [..., q]
    """
    q2 = z.size(-1)
    if q2 % 2 != 0:
        raise ValueError(f"Expected even last dim for packed complex, got {q2}")
    zc = torch.view_as_complex(z.view(*z.shape[:-1], q2 // 2, 2).contiguous())
    n = torch.linalg.norm(zc, dim=-1, keepdim=True).clamp_min(eps)
    zc = zc / n
    return torch.view_as_real(zc).reshape_as(z)


def _fubini_study_distance(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    q: [B, 2q] packed complex
    p: [B, K, 2q] packed complex
    returns: [B, K]
    """
    qc = torch.view_as_complex(q.view(q.size(0), -1, 2).contiguous())
    pc = torch.view_as_complex(p.view(p.size(0), p.size(1), -1, 2).contiguous())
    qc = qc / (torch.linalg.norm(qc, dim=-1, keepdim=True).clamp_min(eps))
    pc = pc / (torch.linalg.norm(pc, dim=-1, keepdim=True).clamp_min(eps))
    ip = (qc.unsqueeze(1).conj() * pc).sum(-1).abs().clamp(0.0, 1.0 - eps)
    return torch.arccos(ip)


def _poincare_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, D], y: [B, K, D], all inside unit ball.
    returns: [B, K]
    """
    x = x.clamp(-0.999, 0.999)
    y = y.clamp(-0.999, 0.999)
    x2 = (x * x).sum(-1, keepdim=True).clamp_max(1.0 - 1e-5)  # [B,1]
    y2 = (y * y).sum(-1).clamp_max(1.0 - 1e-5)  # [B,K]
    diff2 = ((x.unsqueeze(1) - y) ** 2).sum(-1)  # [B,K]
    den = ((1.0 - x2) * (1.0 - y2)).clamp_min(eps)
    z = 1.0 + 2.0 * diff2 / den
    return torch.acosh(z.clamp_min(1.0 + eps))


class ConsolidatedLexicon(nn.Module):
    """
    Sense-aware consolidated parameter store on H x CP^q x R factors.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        senses: int = 3,
        d_hyper: int = 32,
        q_complex: int = 8,
        d_euclid: int = 16,
        d_pron: int = 12,
        d_char: int = 16,
        conformal_b: float = 0.02,
        w_h: float = 0.6,
        w_p: float = 0.25,
        w_e: float = 0.15,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.k = int(senses)
        self.dh = int(d_hyper)
        self.q = int(q_complex)
        self.de = int(d_euclid)
        self.dp = int(d_pron)
        self.dc = int(d_char)
        self.context_dim = int(context_dim if context_dim is not None else model_dim)
        self.feature_dim = self.dh + 2 * self.q + self.de + self.dp + self.de + self.dc

        self.mu_h = nn.Parameter(torch.zeros(self.vocab_size, self.k, self.dh))
        self.mu_p = nn.Parameter(
            _unit_complex_real(torch.randn(self.vocab_size, self.k, 2 * self.q) / math.sqrt(self.q))
        )
        self.mu_e = nn.Parameter(torch.zeros(self.vocab_size, self.k, self.de))
        self.pron = nn.Parameter(torch.zeros(self.vocab_size, self.k, self.dp))
        self.morph = nn.Parameter(torch.zeros(self.vocab_size, self.k, self.de))
        self.char = nn.Parameter(torch.zeros(self.vocab_size, self.k, self.dc))
        self.sense_logit = nn.Parameter(torch.zeros(self.vocab_size, self.k))

        self.w = nn.Parameter(torch.tensor([float(w_h), float(w_p), float(w_e)]))
        self.b = nn.Parameter(torch.tensor(float(conformal_b)), requires_grad=False)

        self.cue_h = nn.Linear(self.model_dim, self.dh)
        self.cue_p = nn.Linear(self.model_dim, 2 * self.q)
        self.cue_e = nn.Linear(self.model_dim, self.de)
        self.gate = nn.Linear(self.context_dim, self.k)
        self.phi = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, 2 * self.model_dim),
            nn.GELU(),
            nn.Linear(2 * self.model_dim, self.model_dim),
        )

    @torch.no_grad()
    def renorm_constraints_(self):
        self.mu_h.data.clamp_(-0.95, 0.95)
        self.mu_p.data = _unit_complex_real(self.mu_p.data)

    def _weighted_structured(self, idx: torch.Tensor, sense_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        take = lambda m: m.index_select(0, idx)
        mu_h = (sense_w.unsqueeze(-1) * take(self.mu_h)).sum(dim=1)
        mu_p = (sense_w.unsqueeze(-1) * take(self.mu_p)).sum(dim=1)
        mu_e = (sense_w.unsqueeze(-1) * take(self.mu_e)).sum(dim=1)
        pron = (sense_w.unsqueeze(-1) * take(self.pron)).sum(dim=1)
        morph = (sense_w.unsqueeze(-1) * take(self.morph)).sum(dim=1)
        char = (sense_w.unsqueeze(-1) * take(self.char)).sum(dim=1)
        return {
            "mu_h": mu_h,
            "mu_p": mu_p,
            "mu_e": mu_e,
            "pron": pron,
            "morph": morph,
            "char": char,
            "packed": torch.cat([mu_h, mu_p, mu_e, pron, morph, char], dim=-1),
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        base_embed: torch.Tensor,
        context_feat: Optional[torch.Tensor] = None,
    ):
        """
        token_ids: [N] long
        base_embed: [N, D]
        context_feat: [N, C] optional
        returns: fused [N, D], sense_weights [N, K], aux dict
        """
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be [N], got {list(token_ids.shape)}")
        if base_embed.dim() != 2 or base_embed.size(-1) != self.model_dim:
            raise ValueError(f"base_embed must be [N,{self.model_dim}], got {list(base_embed.shape)}")
        if context_feat is None:
            context_feat = base_embed
        if context_feat.dim() != 2 or context_feat.size(0) != token_ids.size(0):
            raise ValueError("context_feat must match batch size of token_ids")

        idx = token_ids
        mu_h = self.mu_h.index_select(0, idx)
        mu_p = self.mu_p.index_select(0, idx)
        mu_e = self.mu_e.index_select(0, idx)

        cue_h = torch.tanh(self.cue_h(base_embed))
        cue_h = cue_h.clamp(-0.95, 0.95)
        cue_p = _unit_complex_real(self.cue_p(base_embed))
        cue_e = self.cue_e(base_embed)

        d_h = _poincare_distance(cue_h, mu_h)
        d_p = _fubini_study_distance(cue_p, mu_p)
        d_e = torch.cdist(cue_e.unsqueeze(1), mu_e, p=2).squeeze(1)

        w = torch.softmax(self.w, dim=0)
        d2 = w[0] * d_h.square() + w[1] * d_p.square() + w[2] * d_e.square()

        agg = torch.cat(
            [
                mu_h.mean(dim=1),
                mu_p.mean(dim=1),
                mu_e.mean(dim=1),
                self.pron.index_select(0, idx).mean(dim=1),
                self.morph.index_select(0, idx).mean(dim=1),
                self.char.index_select(0, idx).mean(dim=1),
            ],
            dim=-1,
        )
        warp = torch.exp(self.b * self.phi(agg)).clamp(0.95, 1.05).squeeze(-1)
        d2 = d2 * warp.unsqueeze(-1)

        prior = self.sense_logit.index_select(0, idx)
        gate = self.gate(context_feat)
        sense_scores = -d2 + prior + gate
        sense_w = torch.softmax(sense_scores, dim=-1)

        structured = self._weighted_structured(idx, sense_w)
        fused = base_embed + self.proj(structured["packed"])
        aux = {
            "dist_h": d_h,
            "dist_p": d_p,
            "dist_e": d_e,
            "warp": warp,
            "weights": sense_w,
            "cue_h": cue_h,
            "cue_p": cue_p,
            "cue_e": cue_e,
            "structured": structured,
        }
        return fused, sense_w, aux

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "manifold_semantic": [self.mu_h, self.mu_p, self.mu_e, self.w],
            "associated_features": [self.pron, self.morph, self.char, self.sense_logit],
            "routing_heads": list(self.cue_h.parameters())
            + list(self.cue_p.parameters())
            + list(self.cue_e.parameters())
            + list(self.gate.parameters())
            + list(self.phi.parameters())
            + list(self.proj.parameters()),
        }

    def build_optimizer_param_groups(
        self,
        base_lr: float,
        assoc_lr_scale: float = 1.0,
        routing_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ):
        groups = self.get_parameter_groups()
        return [
            {"params": groups["manifold_semantic"], "lr": base_lr, "weight_decay": weight_decay},
            {
                "params": groups["associated_features"],
                "lr": base_lr * assoc_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["routing_heads"],
                "lr": base_lr * routing_lr_scale,
                "weight_decay": weight_decay,
            },
        ]
