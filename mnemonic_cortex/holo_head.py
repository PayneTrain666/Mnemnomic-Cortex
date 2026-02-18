import math

import torch
import torch.nn as nn


def _unit_complex(z: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    n = torch.linalg.norm(z, dim=-1, keepdim=True) + eps
    return z / n


def _fs_distance(q: torch.Tensor, phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # q:[B,D] complex, phi:[B,K,D] complex -> [B,K]
    ip = torch.sum(q.unsqueeze(1).conj() * phi, dim=-1).abs().clamp(0.0, 1.0 - eps)
    return torch.arccos(ip)


def _cconv_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = torch.fft.rfft(a.float(), dim=-1)
    bf = torch.fft.rfft(b.float(), dim=-1)
    cf = af * bf
    c = torch.fft.irfft(cf, n=a.size(-1), dim=-1)
    return c.to(a.dtype)


class HoloHead(nn.Module):
    """
    Quantum-holographic head with complex slots + FS lookup.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_heads: int = 1,
        temperature: float = 1.0,
        phase_noise_std: float = 0.0,
        lightbulb_thresh: float = 0.92,
        explosive_temp: float = 0.6,
        explosive_alpha: float = 0.6,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("phase_noise_std", torch.tensor(float(phase_noise_std)))
        self.register_buffer("lightbulb_thresh", torch.tensor(float(lightbulb_thresh)))
        self.register_buffer("explosive_temp", torch.tensor(float(explosive_temp)))
        self.register_buffer("explosive_alpha", torch.tensor(float(explosive_alpha)))

        real = torch.randn(num_heads, num_slots, dim) / math.sqrt(dim)
        imag = torch.randn(num_heads, num_slots, dim) / math.sqrt(dim)
        self.slots = nn.Parameter(torch.complex(real, imag))

        self.q_enc = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 2 * dim),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        with torch.no_grad():
            self.slots.copy_(_unit_complex(self.slots))

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature.fill_(float(t))

    @torch.no_grad()
    def set_phase_noise(self, s: float):
        self.phase_noise_std.fill_(float(s))

    @torch.no_grad()
    def set_lightbulb(self, thresh: float, explosive_temp: float, explosive_alpha: float):
        self.lightbulb_thresh.fill_(float(thresh))
        self.explosive_temp.fill_(float(explosive_temp))
        self.explosive_alpha.fill_(float(explosive_alpha))

    @torch.no_grad()
    def renorm_slots(self):
        self.slots.copy_(_unit_complex(self.slots))

    def _encode_complex(self, q_real: torch.Tensor) -> torch.Tensor:
        if q_real.size(-1) != self.dim:
            raise ValueError(f"HoloHead expects dim={self.dim}, got {q_real.size(-1)}")
        z = self.q_enc(q_real)
        z = z.reshape(z.size(0), 2, self.dim).permute(0, 2, 1).contiguous()
        zc = torch.view_as_complex(z)
        if self.training and float(self.phase_noise_std.item()) > 0:
            noise = torch.randn_like(zc.real) * float(self.phase_noise_std.item())
            zc = torch.complex(zc.real, zc.imag + noise)
        return _unit_complex(zc)

    def weights(self, q_feat_real: torch.Tensor, slot_indices: torch.Tensor, head_id: int = 0, override_temp=None):
        bsz, k = slot_indices.shape
        qz = self._encode_complex(q_feat_real)
        phi = self.slots[head_id].index_select(0, slot_indices.reshape(-1)).view(bsz, k, self.dim)
        d_fs = _fs_distance(qz, phi)

        temp = float(self.temperature.item()) if override_temp is None else float(override_temp)
        logits = -d_fs / max(1e-6, temp)
        w = torch.softmax(logits, dim=-1)

        resonance = w.max(dim=-1).values
        coh_vec = torch.sum(w.unsqueeze(-1) * phi, dim=1)
        coherence = torch.linalg.norm(coh_vec, dim=-1).real
        alpha = self.alpha_head(q_feat_real)
        return {
            "weights": w,
            "fs_dist": d_fs,
            "resonance": resonance,
            "coherence": coherence,
            "alpha": alpha,
        }

    def bind_roles_fillers(self, roles_real: torch.Tensor, fillers_real: torch.Tensor) -> torch.Tensor:
        return _cconv_bind(roles_real, fillers_real)

    def unbind(self, code_real: torch.Tensor, role_real: torch.Tensor) -> torch.Tensor:
        cf = torch.fft.rfft(code_real.float(), dim=-1)
        rf = torch.conj(torch.fft.rfft(role_real.float(), dim=-1))
        vf = cf * rf
        v = torch.fft.irfft(vf, n=code_real.size(-1), dim=-1)
        return v.to(code_real.dtype)
