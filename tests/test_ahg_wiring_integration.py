import tempfile
from pathlib import Path

import torch

from mnemonic_cortex.ahg import AHGConfig, AntiHallucinationGuard
from mnemonic_cortex.config_loader import load_yaml_config
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_ahg_config_validation_rejects_invalid_thresholds():
    cfg = AHGConfig(phase_rho=1.5)
    try:
        cfg.validate()
        assert False, "expected ValueError for invalid phase_rho"
    except ValueError:
        pass


def test_config_loader_parses_string_booleans_for_ahg():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(
            "\n".join(
                [
                    "stores: {}",
                    "ahg:",
                    "  ask_on_uncertain: \"false\"",
                    "  refuse_on_high_risk: \"true\"",
                ]
            ),
            encoding="utf-8",
        )
        cfg = load_yaml_config(str(p))
    assert cfg.ahg.ask_on_uncertain is False
    assert cfg.ahg.refuse_on_high_risk is True


def test_ahg_decide_handles_non_finite_signal_payload():
    guard = AntiHallucinationGuard(AHGConfig())
    decision = guard.decide(
        broker_result={"signals": {"proto_distance": float("nan"), "phase_agreement": float("inf")}},
        cross_diag={"agreement": float("nan")},
    )
    assert decision.action in {"explosive", "allow", "refine", "ask"}


def test_cortex_ahg_wiring_sets_last_decision_when_broker_enabled():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16, cms_vocab_size=64, cms_senses=2)
    model.enable_consolidation_broker(vocab_size=64)
    x = torch.randn(2, 4, 16)
    ctx = torch.randn(2, 16)
    _ = model.retrieve_memory(x, ctx, strategy="direct", query_token_ids=torch.randint(0, 64, (2, 4)))
    assert isinstance(model.last_ahg_decision, dict)
    assert model.last_ahg_decision.get("action") in {"explosive", "allow", "refine", "ask"}
