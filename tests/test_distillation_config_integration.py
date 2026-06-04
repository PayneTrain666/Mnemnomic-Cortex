import tempfile
from pathlib import Path

import torch

from mnemonic_cortex.config_loader import load_yaml_config
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_config_loader_parses_distillation_section():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(
            "\n".join(
                [
                    "stores: {}",
                    "ahg: {}",
                    "distill:",
                    "  enabled: \"true\"",
                    "  embedding_weight: 0.7",
                    "  mse_weight: 0.05",
                    "  neighbor_kl_weight: 0.3",
                    "  cms_teacher_weight: 0.1",
                    "  teacher_domain: core",
                    "  student_domains: [reasoning, science]",
                    "  neighbor_k: 12",
                    "  sim_temp: 0.2",
                ]
            ),
            encoding="utf-8",
        )
        cfg = load_yaml_config(str(p))
    assert cfg.distill.enabled is True
    assert cfg.distill.teacher_domain == "core"
    assert tuple(cfg.distill.student_domains) == ("reasoning", "science")
    assert cfg.distill.neighbor_k == 12
    assert abs(cfg.distill.sim_temp - 0.2) < 1e-9


def test_cortex_configure_distillation_wires_advanced_distiller():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_advanced_consolidation(cps_domains=["core", "reasoning"])
    model.configure_distillation(
        {
            "enabled": True,
            "teacher_domain": "core",
            "student_domains": ("reasoning",),
            "neighbor_k": 10,
            "sim_temp": 0.15,
        }
    )
    assert model.distillation_config.enabled is True
    assert model.advanced_distiller is not None
    assert model.advanced_distiller.distill_config.enabled is True
    assert model.advanced_distiller.neighbor_k == 10
    assert abs(model.advanced_distiller.sim_temp - 0.15) < 1e-9


def test_process_aux_losses_include_distillation_when_enabled():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_advanced_consolidation(cps_domains=["core", "reasoning"])
    model.configure_distillation(
        {
            "enabled": True,
            "teacher_domain": "core",
            "student_domains": ("reasoning",),
            "embedding_weight": 0.6,
            "mse_weight": 0.1,
            "neighbor_kl_weight": 0.2,
            "cms_teacher_weight": 0.1,
        }
    )
    x = torch.randn(2, 4, 16)
    ctx = torch.randn(2, 16)
    token_ids = torch.randint(0, 32, (2, 4))
    _, aux = model(
        x,
        ctx,
        operation="process",
        return_aux_losses=True,
        token_ids=token_ids,
    )
    assert "distill_loss" in aux
    assert torch.is_tensor(aux["distill_loss"])
    assert aux["distill_loss"].dim() == 0
    assert torch.isfinite(aux["distill_loss"])
