import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.hypergraph_manifold import MutationDirection, MutationToken


def _tokens():
    return [
        MutationToken(
            token_id="tok-1",
            variable_id="var-a",
            direction=MutationDirection.POSITIVE,
            magnitude_bin_id="mag-1",
            probability=0.7,
        ),
        MutationToken(
            token_id="tok-2",
            variable_id="var-b",
            direction=MutationDirection.NEGATIVE,
            magnitude_bin_id="mag-2",
            probability=0.6,
        ),
        MutationToken(
            token_id="tok-3",
            variable_id="var-a",
            direction=MutationDirection.NEUTRAL,
            magnitude_bin_id="mag-2",
            probability=0.5,
        ),
    ]


def test_cortex_hgm_bridge_can_be_enabled_and_run():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16, hgm_enabled=True)
    out = model.run_hypergraph_manifold(_tokens(), top_k=4)
    assert out["expansion"].validation.ok is True
    assert out["hgm1"].validation.ok is True
    assert out["hgm2"].validation.ok is True
    assert len(out["hgm2"].routing.assignments) > 0
    metrics = model.get_metrics()
    assert metrics["hgm_enabled"] == 1.0
    assert metrics["hgm_assignments"] > 0


def test_cortex_hgm_bridge_disabled_path_is_safe():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16, hgm_enabled=False)
    out = model.run_hypergraph_manifold(_tokens(), top_k=3)
    assert out["enabled"] is False
    assert "reason" in out
