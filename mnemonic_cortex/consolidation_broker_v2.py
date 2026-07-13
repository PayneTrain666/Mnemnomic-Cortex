"""
Plain-language summary
----------------------
What this file is for: Newer consolidation broker with clearer config.
How it fits in the system: Coordinates multi-store consolidation in modern builds.
Status: ACTIVE when consolidation enabled
Important notes for non-coders: Preferred broker implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conflict_resolver import ConflictResolver
from .multi_cps import MultiCPSManager


@dataclass
class BrokerCfg:
    alpha_base: float = 0.2
    alpha_cap: float = 0.6
    alpha_min: float = 0.05
    confidence_gate: float = 0.25
    pull_cps_weight: float = 0.15
    push_cps_weight: float = 0.10
    agree_weight: float = 1e-3


class ConsolidationBrokerV2(nn.Module):
    """
    Advanced CPS<->CMS broker with optional multi-domain CPS routing.
    """

    def __init__(
        self,
        cms,
        cps=None,
        cps_fuser=None,
        cfg: BrokerCfg = BrokerCfg(),
        multi_cps: Optional[MultiCPSManager] = None,
    ):
        super().__init__()
        self.cms = cms
        self.cps = cps
        self.cps_fuser = cps_fuser
        self.multi_cps = multi_cps
        self.cfg = cfg
        if self.multi_cps is None and (self.cps is None or self.cps_fuser is None):
            raise ValueError("Provide either multi_cps or (cps and cps_fuser)")
        d = cms.default_cfg.d_model
        self.merge_gate = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.conflicts = ConflictResolver(cms)

    def _select_cps(self, key: str, domain: Optional[str] = None):
        if self.multi_cps is not None:
            dom = domain or self.multi_cps.route(key)
            cps, fuser = self.multi_cps.get(dom)
            return dom, cps, fuser
        if self.cps is None or self.cps_fuser is None:
            raise ValueError("CPS selection failed: no multi_cps and no default (cps, cps_fuser)")
        return domain or "default", self.cps, self.cps_fuser

    def compute_alpha(self, importance: torch.Tensor) -> float:
        v = float(importance.detach().clamp(0, 1).mean().item())
        a = self.cfg.alpha_base * (0.5 + v)
        return max(self.cfg.alpha_min, min(self.cfg.alpha_cap, a))

    @torch.no_grad()
    def ingest_from_ltm(self, key: str, candidate_view: Dict[str, torch.Tensor], importance: torch.Tensor, src_info: Optional[Dict] = None):
        e = candidate_view.get("E", None)
        if e is not None:
            g = float(self.merge_gate(e.unsqueeze(0)).item())
            if g < self.cfg.confidence_gate:
                return
        alpha = self.compute_alpha(importance)
        if e is not None:
            _ = self.conflicts.update_on_merge(key, e)
            alpha = alpha * (1.0 - self.conflicts.penalty(key))
        self.cms.write(key, candidate_view, alpha=alpha, src_info=src_info)

    @torch.no_grad()
    def cms_pull_to_cps(self, key: str, domain: Optional[str] = None):
        dom, cps, _ = self._select_cps(key, domain)
        _ = dom
        if key not in self.cms.keys():
            return
        unit = self.cms.get(key)
        up = cps.ensure(key)
        uv = up.view()
        mv = unit.view()
        w = self.cfg.pull_cps_weight
        if "E" in uv and "E" in mv:
            up.euclid.copy_((1 - w) * up.euclid + w * mv["E"])
        if "H" in uv and "H" in mv:
            up.hyp.copy_((1 - w) * up.hyp + w * mv["H"])
        if "S" in uv and "S" in mv:
            up.spher.copy_((1 - w) * up.spher + w * mv["S"])
        if "F" in uv and "F" in mv:
            mu, lv = uv["F"]
            cmu, clv = mv["F"]
            up.fisher_mu.copy_((1 - w) * mu + w * cmu)
            up.fisher_lv.copy_(torch.logaddexp((1 - w) * lv, w * clv))
        if "T" in uv and "T" in mv:
            up.torus.copy_((1 - w) * up.torus + w * mv["T"])
        if "P" in uv and "P" in mv:
            amp, phi = uv["P"]
            camp, cphi = mv["P"]
            up.phase_amp.copy_((1 - w) * amp + w * camp)
            up.phase_phi.copy_(phi + w * ((cphi - phi + torch.pi) % (2 * torch.pi) - torch.pi))
        up.project_after_step()

    @torch.no_grad()
    def cps_push_to_cms(self, key: str, domain: Optional[str] = None):
        dom, cps, _ = self._select_cps(key, domain)
        _ = dom
        up = cps.ensure(key)
        self.cms.ensure(key).ema_merge(up.view(), alpha=self.cfg.push_cps_weight, src_info={"src": "cps_push"})

    def cohesion_regularizer(self, keys: List[str], domain: Optional[str] = None):
        if not keys:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = 0.0
        count = 0
        for k in keys:
            dom, cps, fuser = self._select_cps(k, domain)
            _ = dom
            if k not in cps._registry or k not in self.cms.keys():
                continue
            fused_cms = self.cms.read(k)
            fused_cps, _ = fuser.fuse(cps.get(k).view())
            total = total + F.mse_loss(fused_cms, fused_cps)
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.cfg.agree_weight * (total / count)

