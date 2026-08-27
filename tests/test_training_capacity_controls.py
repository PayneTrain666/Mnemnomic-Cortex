from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from mnemonic_cortex.config import CortexConfig
from mnemonic_cortex.config_loader import load_unified_yaml_config
from tools.babi_train_eval import (
    _configure_sdpa_backends as configure_babi_sdpa,
    _resolve_amp_dtype as resolve_babi_amp_dtype,
    build_arg_parser,
)
from tools.copy_task_gpu_train import (
    _configure_sdpa_backends as configure_copy_sdpa,
    _resolve_amp_dtype as resolve_copy_amp_dtype,
)
from tools.count_model_params import literal_training_bytes


def test_capacity_profile_is_opt_in_and_preserves_defaults():
    default = CortexConfig(capacity_profile="compact")
    applied = CortexConfig(
        capacity_profile="compact",
        apply_capacity_profile=True,
    )

    assert default.wm_slot_dim == 256
    assert default.hg_mem_slots == 1028
    assert applied.wm_slot_dim == 192
    assert applied.hg_mem_slots == 768


def test_unified_capacity_profile_opt_in_overrides_sizing_fields():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "profile.yaml"
        path.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  capacity_profile: compact",
                    "  apply_capacity_profile: true",
                    "  wm_slot_dim: 999",
                    "stores: {}",
                    "ahg: {}",
                ]
            ),
            encoding="utf-8",
        )
        unified = load_unified_yaml_config(str(path))

    assert unified.cortex.apply_capacity_profile is True
    assert unified.cortex.wm_slot_dim == 192
    assert unified.cortex.ltm_hg_slots == 768


def test_literal_training_bytes_keeps_categories_separate():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()

    report = literal_training_bytes(model, optimizer=optimizer)
    expected_parameters = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    expected_gradients = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    assert report["parameter_bytes"] == expected_parameters
    assert report["gradient_bytes"] == expected_gradients
    assert report["optimizer_bytes"] > 0
    assert report["cps_rollback_bytes"] is None
    assert report["cuda_peak_allocated_bytes"] is None


def test_amp_dtype_defaults_and_cpu_safety():
    args = build_arg_parser().parse_args([])
    assert args.amp_dtype == "auto"
    assert args.report_capacity is False

    for resolver in (resolve_babi_amp_dtype, resolve_copy_amp_dtype):
        enabled, dtype, name = resolver(torch.device("cpu"), True, "bf16")
        assert enabled is False
        assert dtype == torch.float16
        assert name == "disabled"


def test_sdpa_configuration_skips_non_cuda():
    for configure in (configure_babi_sdpa, configure_copy_sdpa):
        report = configure(torch.device("cpu"))
        assert report["status"] == "skipped_non_cuda"

