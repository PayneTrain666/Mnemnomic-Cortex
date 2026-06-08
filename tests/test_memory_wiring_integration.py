import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory
from benchmark.models import CortexSeqModel


def test_cortex_process_writes_all_ltm_banks():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, wm_slots=8, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.train()
    x = torch.randn(2, 4, 32)
    ctx = torch.randn(2, 32)
    out = cortex(x, context=ctx, operation="process")
    assert out.shape == (2, 4, 32)
    assert float(cortex.long_term_memory.hg.usage_counts.sum().item()) > 0.0
    assert float(cortex.long_term_memory.cgmn.usage_counts.sum().item()) > 0.0
    assert float(cortex.long_term_memory.curved.usage_counts.sum().item()) > 0.0


def test_enable_cps_cms_full_stack_wires_optional_subsystems():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32)
    cortex.enable_cps_cms_full_stack(
        vocab_size=64,
        enable_broker=True,
        enable_advanced=False,
        enable_reasoning_bridge=True,
        enable_qdt_wm_bridge=True,
    )
    assert cortex.consolidated_lexicon is not None
    assert cortex.consolidation_broker is not None
    assert cortex.shared_memory_subsystem is not None
    assert cortex.hg_episodic_ltm is not None
    assert hasattr(cortex.working_memory, "get_attention_stack_output")


def test_cortex_seq_model_accepts_train_script_kwargs():
    model = CortexSeqModel(
        vocab_size=32,
        d_model=32,
        ltm_hg_slots=32,
        ltm_cgmn_slots=32,
        ltm_curved_slots=16,
        ltm_curved_hidden_dim=48,
        ltm_curved_hidden_mult=1.5,
        hgm_enabled=True,
        cms_enabled=False,
    )
    assert model.ltm_curved_hidden_dim == 48
    assert model.hgm_enabled is True
    x = torch.randint(0, 32, (2, 5))
    logits = model(x)
    assert logits.shape == (2, 5, 32)


def test_episodic_store_auto_syncs_to_triple_hybrid():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.enable_shared_memory_subsystem(num_slots=64, num_systems=4)
    cortex.enable_hg_episodic_ltm()
    vectors = torch.randn(3, 32)
    cortex.store_episodic_trace(
        episode_id="ep-1",
        episode_vectors=vectors,
        step_range=(0, 2),
    )
    assert float(cortex.long_term_memory.hg.usage_counts.sum().item()) > 0.0
    assert float(cortex.long_term_memory.cgmn.usage_counts.sum().item()) > 0.0
    assert float(cortex.long_term_memory.curved.usage_counts.sum().item()) > 0.0


def test_triple_hybrid_ltm_adapter_reads_live_banks():
    from mnemonic_cortex.working_memory.wm_triple_hybrid_ltm_adapter import TripleHybridLTMExternalMemoryBank
    from mnemonic_cortex.working_memory.wm_external_memory_interfaces import ExternalMemoryQuery

    ltm = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=32, cgmn_slots=32, curved_slots=16)
    x = torch.randn(2, 4, 32)
    ltm(x, operation="write")
    adapter = TripleHybridLTMExternalMemoryBank(dim=32, triple_hybrid=ltm)
    query = ExternalMemoryQuery("ltm", query_state=x.mean(dim=1), context=x)
    response = adapter.query(query, top_k=4)
    assert response.metadata["adapter_kind"] == "triple_hybrid_ltm_adapter"
    assert response.memory_state.shape == (2, 4, 32)
    assert float(response.scores.mean().item()) != 0.0


def test_qdt_ltm_adapter_wires_through_cortex_full_stack():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.enable_cps_cms_full_stack(
        vocab_size=64,
        enable_broker=False,
        enable_advanced=False,
        enable_reasoning_bridge=False,
        enable_qdt_wm_bridge=True,
    )
    qdt = cortex.working_memory.qdt_working_memory
    bank = qdt.dual_fusion.ltm.external_bank
    assert bank.is_attached is True
    assert bank.triple_hybrid is cortex.long_term_memory


def test_qdt_process_uses_live_ltm_during_dual_fusion():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.enable_cps_cms_full_stack(
        vocab_size=64,
        enable_broker=False,
        enable_reasoning_bridge=False,
        enable_qdt_wm_bridge=True,
    )
    cortex.long_term_memory(torch.randn(1, 3, 32), operation="write")
    x = torch.randn(2, 4, 32)
    out = cortex.working_memory(x, operation="process")
    bank = cortex.working_memory.qdt_working_memory.dual_fusion.ltm.external_bank
    assert bank.is_attached is True
    assert "triple_hybrid_ltm" in bank.last_query_trace.get("trace_type", "")
    assert out.shape == x.shape


def test_prefusion_specialization_runs_before_fusion():
    model = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=32, cgmn_slots=32, curved_slots=16)
    x = torch.randn(2, 4, 32)
    model(x, operation="write")
    fused = model(x, operation="read")
    stats = model.last_prefusion_specialization_stats
    assert fused.shape == x.shape
    assert "hg_specialization_mean" in stats
    assert "cgmn_specialization_mean" in stats
    assert "curved_specialization_mean" in stats
    assert 0.0 <= stats["hg_specialization_mean"] <= 1.0


def test_triple_hybrid_unified_constructor_wires_bank_dims():
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        hg_dim=20,
        hg_slots=40,
        hg_qubits=6,
        cgmn_dim=12,
        cgmn_slots=24,
        cgmn_slot_dim=64,
        curved_hidden=48,
        curved_curvature=4,
        curved_slots=16,
        n_transformer_layers=3,
        n_heads=8,
        attention_type="multiscale",
    )
    assert model.hg.D == 20
    assert model.hg.M == 40
    assert model.hg.Q == 6
    assert model.cgmn.D == 12
    assert model.cgmn.H == 64
    assert model.curved.H == 48
    assert model.attention_type == "multiscale"
    from mnemonic_cortex.memory_attention import MultiScaleAttention
    from mnemonic_cortex.memory_transformer_v2 import (
        EnhancedCurvedMemoryWithTransformerV2,
        EnhancedHyperGeometricMemoryWithTransformerV2,
    )

    assert isinstance(model.hyper_geometric, EnhancedHyperGeometricMemoryWithTransformerV2)
    assert isinstance(model.curved, EnhancedCurvedMemoryWithTransformerV2)
    assert model.hg is model.hyper_geometric
    assert isinstance(model.prefusion_compare_attn, MultiScaleAttention)


def test_triple_hybrid_slot_capacity_propagates_to_bank_buffers():
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        hg_slots=48,
        cgmn_slots=40,
        curved_slots=24,
        spatial_slots=20,
        depth_profile="compact",
    )
    assert model.hg.M == 48
    assert model.hg.q_memory_slots.size(0) == 48
    assert model.cgmn.M == 40
    assert model.cgmn.memory_slots.size(0) == 40
    assert model.curved.M == 24
    assert model.curved.memory_slots.size(0) == 24
    assert model.spatial_ltm is not None
    assert model.spatial_ltm.slots == 20


def test_hns_fusion_stack_wires_router_and_lightbulb():
    model = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=32, cgmn_slots=32, curved_slots=16)
    x = torch.randn(2, 4, 32)
    model(x, operation="write")
    fused = model(x, operation="read")
    assert fused.shape == x.shape
    assert model.enable_hns_fusion is True
    assert model.fusion_in_dim == 32 * 6
    assert "lightbulb_intensity" in model.last_router_stats
    assert len(model.lightbulb_logger.recent(4)) >= 1


def test_qdt_wrapper_accepts_external_attention_context():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.enable_cps_cms_full_stack(
        vocab_size=64,
        enable_broker=False,
        enable_reasoning_bridge=False,
        enable_qdt_wm_bridge=True,
    )
    ctx = torch.randn(2, 4, 32)
    cortex.working_memory.set_external_attention_context(ctx)
    qdt = cortex.working_memory.qdt_working_memory
    assert qdt.external_attention_context is not None
    assert tuple(qdt.external_attention_context.shape) == (2, 4, 32)


def test_read_banks_matches_forward_fused_output():
    model = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=32, cgmn_slots=32, curved_slots=16)
    x = torch.randn(2, 4, 32)
    model(x, operation="write")
    fused = model(x, operation="read")
    banks = model.read_banks(x, include_fused=True)
    assert tuple(banks["fused"].shape) == tuple(fused.shape)
    assert tuple(banks["hg"].shape) == tuple(x.shape)


def test_retrieve_memory_uses_qdt_process_operation():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32, ltm_hg_slots=32, ltm_cgmn_slots=32, ltm_curved_slots=16)
    cortex.enable_cps_cms_full_stack(
        vocab_size=64,
        enable_broker=False,
        enable_reasoning_bridge=False,
        enable_qdt_wm_bridge=True,
    )
    x = torch.randn(2, 4, 32)
    ctx = torch.randn(2, 32)
    out = cortex.retrieve_memory(x, ctx, strategy="direct")
    assert out.shape == (2, 32)
    assert cortex._wm_read_operation() == "process"


def test_hgm_enabled_wires_episodic_ltm_in_seq_model():
    model = CortexSeqModel(vocab_size=32, d_model=32, hgm_enabled=True, cms_enabled=False)
    assert model.cortex.hg_episodic_ltm is not None


def test_triple_hybrid_read_bank_helpers_match_forward_banks():
    model = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=32, cgmn_slots=32, curved_slots=16)
    x = torch.randn(2, 3, 32)
    model(x, operation="write")
    rhg = model.read_bank("hg", x)
    rcg = model.read_bank("cgmn", x)
    rcv = model.read_bank("curved", x)
    rcv_canonical = model.read_bank("curved_associative", x)
    rspcp = model.read_bank("procedural_spcp", x)
    rhg_canonical = model.read_bank("hg_episodic", x)
    rcg_canonical = model.read_bank("cgmn_semantic", x)
    rspatial = model.read_bank("spatial_topological", x)
    assert tuple(rhg.shape) == tuple(x.shape)
    assert tuple(rcg.shape) == tuple(x.shape)
    assert tuple(rcv.shape) == tuple(x.shape)
    assert torch.allclose(rcv, rcv_canonical, atol=1e-5, rtol=1e-5)
    assert tuple(rspcp.shape) == tuple(x.shape)
    assert torch.allclose(rhg, rhg_canonical, atol=1e-5, rtol=1e-5)
    assert torch.allclose(rcg, rcg_canonical, atol=1e-5, rtol=1e-5)
    assert tuple(rspatial.shape) == tuple(x.shape)


def test_memory_geometry_map_mount_and_structured_write_api():
    cortex = EnhancedMnemonicCortex(input_dim=32, output_dim=32)
    cortex.mount_memory_geometry_maps(
        {
            "hg_episodic": ["hyperbolic"] * 8,
            "cgmn_semantic": ["euclidean"] * 8,
            "curved_associative": ["curved"] * 8,
            "spatial_topological": ["spatial_se3"] * 8,
            "procedural_spcp": ["complex_projective"] * 8,
        }
    )
    desc = cortex.describe_memory_system_structure()
    assert desc["banks"]["hg_episodic"]["geometry_map"][0] == "hyperbolic"
    assert desc["banks"]["curved_associative"]["geometry_map"][0] == "curved"
    assert desc["banks"]["procedural_spcp"]["enabled"] is True

    vec = torch.randn(2, 3, 32)
    curved_structured = cortex.structure_memory_entries("curved_associative", vec, tags=["assoc"])
    assert curved_structured["canonical_bank"] == "curved_associative"
    structured = cortex.structure_memory_entries("procedural_spcp", vec, tags=["proc"])
    assert structured["canonical_bank"] == "procedural_spcp"
    assert len(structured["entries"]) == 6
    written = cortex.write_structured_memory("procedural_spcp", vec, write_scale=0.7)
    assert written["written"] is True
