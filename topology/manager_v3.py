from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

EXT_MODES = ["hyperbolic", "spherical", "euclidean", "fractal", "torus", "cp"]


class DynamicTopologyManagerV2:
    """
    Lightweight topology scheduler for geometry blends and curvature mutation.
    """

    def __init__(
        self,
        bank: str = "generic",
        ema_beta: float = 0.9,
        mutation_rate: float = 0.01,
        cooldown_steps: int = 100,
    ):
        self.bank = str(bank)
        self.ema_beta = float(ema_beta)
        self.mutation_rate = float(mutation_rate)
        self.cooldown_steps = int(cooldown_steps)
        self.fitness_ema: Optional[float] = None
        self._steps_since_mut = 0
        self.b_min = 0.02
        self.b_max = 0.10
        self.mode = "euclidean"

    def update_fitness(self, loss_value: float) -> float:
        fitness = 1.0 / (1.0 + float(loss_value))
        if self.fitness_ema is None:
            self.fitness_ema = fitness
        else:
            self.fitness_ema = self.ema_beta * self.fitness_ema + (1.0 - self.ema_beta) * fitness
        return float(self.fitness_ema)

    def pick_mode(self) -> str:
        f = 0.5 if self.fitness_ema is None else float(self.fitness_ema)
        if f > 0.80:
            self.mode = "hyperbolic"
        elif f > 0.60:
            self.mode = "spherical"
        elif f > 0.40:
            self.mode = "euclidean"
        else:
            self.mode = "fractal"
        return self.mode

    def mode_priors(self) -> torch.Tensor:
        # [H,S,E,F,T,CP]
        b = self.bank.lower()
        if b in ("wm", "working", "curved", "curved-wm"):
            return torch.tensor([0.10, 0.24, 0.30, 0.08, 0.22, 0.06], dtype=torch.float32)
        if b in ("hg", "hyper", "hyper_geometric"):
            return torch.tensor([0.38, 0.18, 0.18, 0.10, 0.08, 0.08], dtype=torch.float32)
        if b in ("cgmn", "geo", "geometric"):
            return torch.tensor([0.18, 0.30, 0.28, 0.10, 0.08, 0.06], dtype=torch.float32)
        return torch.tensor([0.20, 0.24, 0.24, 0.12, 0.10, 0.10], dtype=torch.float32)

    def _effective_temperature(
        self, telemetry: Optional[Dict[str, float]] = None, temperature: Optional[float] = None
    ) -> float:
        if temperature is not None:
            return float(max(0.3, min(2.0, temperature)))
        fit = 0.5 if self.fitness_ema is None else float(self.fitness_ema)
        t = 1.3 - 0.6 * fit
        if telemetry:
            ent = float(telemetry.get("entropy", 0.0))
            t *= 1.0 - 0.15 * min(1.0, max(0.0, ent))
        return float(max(0.3, min(2.0, t)))

    def mode_weights(
        self,
        mode_logits: Optional[torch.Tensor] = None,
        telemetry: Optional[Dict[str, float]] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pri = self.mode_priors()
        pri = pri / pri.sum().clamp_min(1e-8)
        if mode_logits is None:
            logits = torch.log(pri.clamp_min(1e-8))
        else:
            p_log = torch.log(pri.to(mode_logits.device, mode_logits.dtype).clamp_min(1e-8))
            logits = mode_logits + p_log
        temp = self._effective_temperature(telemetry=telemetry, temperature=temperature)
        w_ext = F.softmax(logits / max(1e-6, temp), dim=-1)  # (6,)
        w_h = w_ext[0]
        w_s = w_ext[1]
        w_e = w_ext[2]
        w_f = w_ext[3] + 0.5 * w_ext[4] + 0.5 * w_ext[5]
        w4 = torch.stack([w_h, w_s, w_e, w_f], dim=0)
        w4 = w4 / w4.sum().clamp_min(1e-8)
        ext = {m: float(w_ext[i].detach().item()) for i, m in enumerate(EXT_MODES)}
        return w4, ext

    @staticmethod
    def curvature_scalar(curvature_tensor: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        if curvature_tensor.ndim == 1:
            out = curvature_tensor
        elif curvature_tensor.ndim == 2:
            if reduce == "median":
                out = curvature_tensor.median(dim=-1).values
            else:
                out = curvature_tensor.mean(dim=-1)
        else:
            raise ValueError("curvature_tensor must be shape (M,) or (M,D)")
        return out.clamp(-1.0, 1.0)

    @torch.no_grad()
    def mutate_curvature_scalar(self, curv_scalar: torch.Tensor) -> torch.Tensor:
        if self._steps_since_mut < self.cooldown_steps:
            self._steps_since_mut += 1
            return curv_scalar.clamp(-1.0, 1.0)
        rate = float(max(1e-6, self.mutation_rate))
        if self.mode == "hyperbolic":
            delta = rate * torch.randn_like(curv_scalar)
            curv = curv_scalar + delta
        elif self.mode == "spherical":
            delta = rate * torch.abs(torch.randn_like(curv_scalar))
            curv = curv_scalar - 0.5 * delta
        elif self.mode == "euclidean":
            curv = 0.95 * curv_scalar
        else:
            noise = torch.randn_like(curv_scalar)
            kernel = torch.tensor([0.25, 0.5, 0.25], device=curv_scalar.device, dtype=curv_scalar.dtype)
            pad = F.pad(noise.unsqueeze(0).unsqueeze(0), (1, 1), mode="reflect")
            sm = F.conv1d(pad, kernel.view(1, 1, -1)).squeeze(0).squeeze(0)
            curv = curv_scalar + rate * sm
        self._steps_since_mut = 0
        return curv.clamp(-1.0, 1.0)

    @torch.no_grad()
    def mutate_curvature(self, curv_scalar: torch.Tensor, wext: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Gentle slot-wise scalar mutation with optional mode-weighted noise.
        """
        if self._steps_since_mut < self.cooldown_steps:
            self._steps_since_mut += 1
            return curv_scalar.clamp(-1.0, 1.0)
        rate = float(max(1e-6, self.mutation_rate))
        noise = torch.zeros_like(curv_scalar)
        if wext:
            scales = {
                "hyperbolic": 1.00,
                "spherical": 0.75,
                "euclidean": 0.25,
                "fractal": 0.75,
                "torus": 0.50,
                "cp": 0.50,
            }
            for mode, sc in scales.items():
                w = float(wext.get(mode, 0.0))
                if w <= 0.0:
                    continue
                noise = noise + (w * sc) * torch.randn_like(curv_scalar)
            curv = curv_scalar + rate * noise
        else:
            curv = self.mutate_curvature_scalar(curv_scalar)
            self._steps_since_mut = 0
            return curv.clamp(-1.0, 1.0)
        self._steps_since_mut = 0
        return curv.clamp(-1.0, 1.0)

    @torch.no_grad()
    def schedule_conformal_b(
        self, b_current: float, telemetry: Optional[Dict[str, float]] = None
    ) -> float:
        b = float(b_current)
        dist_mean = float(telemetry.get("dist_mean", 1.0)) if telemetry else 1.0
        entropy = float(telemetry.get("entropy", 0.5)) if telemetry else 0.5
        if dist_mean > 1.2 and entropy > 0.6:
            b += 0.01
        if entropy < 0.25:
            b -= 0.01
        fit = 0.5 if self.fitness_ema is None else float(self.fitness_ema)
        b *= 0.9 + 0.2 * (1.0 - fit)
        return float(max(self.b_min, min(self.b_max, b)))

