"""
Plain-language summary
----------------------
What this file is for: Tracks running statistics that decide lightbulb activation.
How it fits in the system: Controller layer above raw lightbulb detectors.
Status: OPT-IN
Important notes for non-coders: Tuning here changes how often 'aha' recall fires.
"""

import math

import torch
import torch.nn as nn


class RunningMoments(nn.Module):
    """EMA mean/var tracker for z-scores."""

    def __init__(self, beta=0.95, eps=1e-6):
        super().__init__()
        self.beta = float(beta)
        self.eps = eps
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("var", torch.tensor(1.0))
        self._init = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float().mean()
        if not self._init:
            self.mean.fill_(x)
            self.var.fill_(1.0)
            self._init = True
            return
        m = float(self.mean.item())
        v = float(self.var.item())
        m_new = self.beta * m + (1.0 - self.beta) * float(x)
        v_new = self.beta * v + (1.0 - self.beta) * float((x - m_new) ** 2)
        self.mean.fill_(m_new)
        self.var.fill_(max(v_new, self.eps))

    @torch.no_grad()
    def zscore(self, x: torch.Tensor):
        x = x.detach().float().mean()
        return (x - self.mean.item()) / math.sqrt(self.var.item() + self.eps)


class LightbulbController(nn.Module):
    """
    Multi-signal detector + staged actuator for explosive recall.
    """

    def __init__(
        self,
        z_thresh_on=2.0,
        z_thresh_off=1.2,
        cooldown_steps=6,
        max_stage2_steps=2,
        budget_per_100=6,
        ema_beta=0.95,
        w_res=0.9,
        w_coh=0.6,
        w_ent=0.7,
        w_marg=0.8,
        prefocus_temp_mult=0.85,
        prefocus_alpha_boost=0.15,
        explosive_temp_mult=0.55,
        explosive_alpha_boost=0.45,
        alpha_max=0.85,
        temp_min=0.45,
    ):
        super().__init__()
        self.m_r = RunningMoments(ema_beta)
        self.m_c = RunningMoments(ema_beta)
        self.m_e = RunningMoments(ema_beta)
        self.m_m = RunningMoments(ema_beta)

        self.z_on = float(z_thresh_on)
        self.z_off = float(z_thresh_off)
        self.cooldown_steps = int(cooldown_steps)
        self.max_stage2_steps = int(max_stage2_steps)
        self.budget_per_100 = int(budget_per_100)

        self.w_res = float(w_res)
        self.w_coh = float(w_coh)
        self.w_ent = float(w_ent)
        self.w_marg = float(w_marg)
        self.prefocus_temp_mult = float(prefocus_temp_mult)
        self.prefocus_alpha_boost = float(prefocus_alpha_boost)
        self.explosive_temp_mult = float(explosive_temp_mult)
        self.explosive_alpha_boost = float(explosive_alpha_boost)
        self.alpha_max = float(alpha_max)
        self.temp_min = float(temp_min)

        self.stage = 0
        self.cooldown = 0
        self.stage2_left = 0
        self.step_count = 0
        self.explosions_in_window = 0

    @torch.no_grad()
    def set_thresholds(self, z_on: float, z_off: float = None):
        self.z_on = float(z_on)
        self.z_off = float(z_off if z_off is not None else max(0.5, self.z_on * 0.6))

    @torch.no_grad()
    def set_explosive(self, explosive_temp_mult: float, explosive_alpha_boost: float):
        self.explosive_temp_mult = float(explosive_temp_mult)
        self.explosive_alpha_boost = float(explosive_alpha_boost)

    @torch.no_grad()
    def set_prefocus(self, prefocus_temp_mult: float, prefocus_alpha_boost: float):
        self.prefocus_temp_mult = float(prefocus_temp_mult)
        self.prefocus_alpha_boost = float(prefocus_alpha_boost)

    @torch.no_grad()
    def set_budget(self, cooldown_steps: int, max_stage2_steps: int, budget_per_100: int):
        self.cooldown_steps = int(cooldown_steps)
        self.max_stage2_steps = int(max_stage2_steps)
        self.budget_per_100 = int(budget_per_100)

    @torch.no_grad()
    def set_limits(self, alpha_max: float, temp_min: float):
        self.alpha_max = float(alpha_max)
        self.temp_min = float(temp_min)

    @torch.no_grad()
    def _score(self, resonance, coherence, weights):
        ent = -torch.sum(weights * (weights.clamp_min(1e-9)).log(), dim=-1).mean()
        top2 = torch.topk(weights, k=2, dim=-1).values
        marg = (top2[:, 0] - top2[:, 1]).mean()

        self.m_r.update(resonance)
        self.m_c.update(coherence)
        self.m_e.update(ent)
        self.m_m.update(marg)

        zr = self.m_r.zscore(resonance)
        zc = self.m_c.zscore(coherence)
        ze = self.m_e.zscore(ent)
        zm = self.m_m.zscore(marg)
        return float(self.w_res * zr + self.w_coh * zc - self.w_ent * ze + self.w_marg * zm)

    @torch.no_grad()
    def forward(self, resonance, coherence, weights):
        self.step_count += 1
        if self.step_count % 100 == 0:
            self.explosions_in_window = 0

        s = self._score(resonance, coherence, weights)
        if self.cooldown > 0:
            self.cooldown -= 1
            if self.stage == 2:
                if self.stage2_left > 0:
                    self.stage2_left -= 1
                if self.stage2_left == 0:
                    self.stage = 1
            return self._actuation()

        if self.stage == 0:
            if s >= self.z_on:
                self.stage = 1
                self.cooldown = self.cooldown_steps
        elif self.stage == 1:
            if s >= self.z_on and self.explosions_in_window < self.budget_per_100:
                self.stage = 2
                self.stage2_left = self.max_stage2_steps
                self.explosions_in_window += 1
                self.cooldown = self.cooldown_steps
            elif s < self.z_off:
                self.stage = 0
        elif self.stage == 2:
            if self.stage2_left > 0:
                self.stage2_left -= 1
            if self.stage2_left == 0 or s < self.z_off:
                self.stage = 1
                self.cooldown = self.cooldown_steps

        return self._actuation()

    @torch.no_grad()
    def _actuation(self):
        if self.stage == 0:
            return dict(stage=0, temp_mult=1.0, alpha_boost=0.0, alpha_max=self.alpha_max)
        if self.stage == 1:
            return dict(
                stage=1,
                temp_mult=max(self.temp_min, self.prefocus_temp_mult),
                alpha_boost=self.prefocus_alpha_boost,
                alpha_max=self.alpha_max,
            )
        return dict(
            stage=2,
            temp_mult=max(self.temp_min, self.explosive_temp_mult),
            alpha_boost=self.explosive_alpha_boost,
            alpha_max=self.alpha_max,
        )
