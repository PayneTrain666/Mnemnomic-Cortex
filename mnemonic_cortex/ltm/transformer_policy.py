"""Dual transformer layer resolution: inherit bank stacks + fixed advantage stacks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualTransformerPolicy:
    """Resolved transformer depths for bank-inherit + fixed advantage stacks."""

    bank_layers: int
    fixed_layers: int
    fusion_layers: int
    decoder_layers: int
    inherited_bank_layers: int
    inherited_fusion_layers: int

    @classmethod
    def resolve(
        cls,
        *,
        bank_layers_cfg: int = 0,
        fixed_layers_cfg: int = 3,
        fusion_layers_cfg: int = 0,
        decoder_layers_cfg: int = 0,
        inherited_bank_layers: int = 3,
        inherited_fusion_layers: int | None = None,
        depth_profile: str = "standard",
    ) -> "DualTransformerPolicy":
        profile = str(depth_profile).strip().lower()
        profile_defaults = {
            "compact": (2, 2),
            "standard": (3, 4),
            "deep": (5, 6),
        }
        default_bank, default_fusion = profile_defaults.get(profile, profile_defaults["standard"])
        inherited_bank = int(inherited_bank_layers) if int(inherited_bank_layers) > 0 else int(default_bank)
        bank = inherited_bank if int(bank_layers_cfg) <= 0 else int(bank_layers_cfg)
        fusion_default = (
            int(inherited_fusion_layers)
            if inherited_fusion_layers is not None
            else int(default_fusion if int(fusion_layers_cfg) <= 0 else max(default_fusion, bank + 1))
        )
        fusion = fusion_default if int(fusion_layers_cfg) <= 0 else int(fusion_layers_cfg)
        decoder = bank if int(decoder_layers_cfg) <= 0 else int(decoder_layers_cfg)
        fixed = max(0, int(fixed_layers_cfg))
        return cls(
            bank_layers=bank,
            fixed_layers=fixed,
            fusion_layers=fusion,
            decoder_layers=decoder,
            inherited_bank_layers=int(inherited_bank),
            inherited_fusion_layers=fusion_default,
        )

    def to_dict(self) -> dict:
        return {
            "bank_layers": self.bank_layers,
            "fixed_layers": self.fixed_layers,
            "fusion_layers": self.fusion_layers,
            "decoder_layers": self.decoder_layers,
            "inherited_bank_layers": self.inherited_bank_layers,
            "inherited_fusion_layers": self.inherited_fusion_layers,
            "dual_stack_active": self.fixed_layers > 0,
        }
