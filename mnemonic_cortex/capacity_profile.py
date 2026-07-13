"""
Plain-language summary
----------------------
What this file is for: Preset size envelopes such as compact, standard, and deep.
How it fits in the system: Quick coherent sets of slot counts and dimensions.
Status: ACTIVE
Important notes for non-coders: Deep profiles need more GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityProfile:
    """Canonical model capacity envelope used across config surfaces."""

    name: str
    input_dim: int
    output_dim: int
    wm_slots: int
    wm_slot_dim: int
    hg_slots: int
    cgmn_slots: int
    curved_slots: int
    spatial_slots: int
    sensory_buffer_size: int
    max_external_context_tokens: int
    max_parameter_tokens: int
    global_hidden_max_layers: int

    @classmethod
    def from_name(cls, name: str) -> "CapacityProfile":
        key = str(name).strip().lower()
        if key == "compact":
            return cls(
                name="compact",
                input_dim=160,
                output_dim=160,
                wm_slots=8,
                wm_slot_dim=192,
                hg_slots=768,
                cgmn_slots=384,
                curved_slots=96,
                spatial_slots=192,
                sensory_buffer_size=8,
                max_external_context_tokens=48,
                max_parameter_tokens=32,
                global_hidden_max_layers=96,
            )
        if key == "deep":
            return cls(
                name="deep",
                input_dim=256,
                output_dim=256,
                wm_slots=10,
                wm_slot_dim=320,
                hg_slots=2048,
                cgmn_slots=1024,
                curved_slots=256,
                spatial_slots=512,
                sensory_buffer_size=16,
                max_external_context_tokens=96,
                max_parameter_tokens=64,
                global_hidden_max_layers=192,
            )
        # standard (default)
        return cls(
            name="standard",
            input_dim=160,
            output_dim=160,
            wm_slots=8,
            wm_slot_dim=256,
            hg_slots=1028,
            cgmn_slots=512,
            curved_slots=128,
            spatial_slots=256,
            sensory_buffer_size=8,
            max_external_context_tokens=64,
            max_parameter_tokens=48,
            global_hidden_max_layers=128,
        )
