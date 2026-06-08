import torch

from geometry.manifold_utils import (
    chart_transform,
    fubini_study_distance,
    poincare_project_ball,
    retract_poincare,
    retract_sphere,
    retract_torus,
    torus_add,
)
from mnemonic_cortex.memory import SharedSlotStore
from mnemonic_cortex.reasoning_depth import (
    LTMDepthAdapter,
    LTMDepthAdapterConfig,
    MANNDepthAdapter,
    MANNDepthAdapterConfig,
    MANNLTMSharedSlotGeometry,
    SharedGeometrySlotConfig,
    SharedDepthSlotRegistry,
)
from mnemonic_cortex.topology_manager import TopologyManagerV3


def _build_shared_geometry(dim: int = 16, *, with_store: bool = False, with_topology: bool = False):
    mann = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12))
    ltm = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=dim, value_dim=dim, slot_count=12))
    store = SharedSlotStore(num_slots=12, slot_dim=dim, num_systems=4, device="cpu", dtype=torch.float32) if with_store else None
    topo = TopologyManagerV3(subsystems=("mann_ltm",)) if with_topology else None
    return MANNLTMSharedSlotGeometry(
        config=SharedGeometrySlotConfig.enabled_default(key_dim=dim, value_dim=dim),
        mann_adapter=mann,
        ltm_adapter=ltm,
        registry=SharedDepthSlotRegistry(),
        shared_slot_store=store,
        topology_manager=topo,
    )


def test_manifold_chart_transform_projects_and_retracts():
    x = torch.randn(2, 8)
    out, meta = chart_transform(x, geometry="hyperbolic", depth_index=2, gain=1.1)
    assert out.shape == x.shape
    assert meta["geom"] == "hyperbolic"
    assert torch.isfinite(out).all()


def test_shared_slot_geometry_uses_manifold_chart_by_default():
    shared = _build_shared_geometry(dim=16)
    query = torch.randn(2, 4, 16)
    out, trace = shared.run_shared_reasoning(
        query,
        content="manifold slot",
        mann_slot_index=1,
        ltm_slot_index=2,
        return_trace=True,
    )
    assert out.shape == (2, 16)
    assert trace["mann_chart_transform"]["manifold_chart"] is True
    assert "manifold_diagnostics" in trace
    assert "distance_mann_mean" in trace["manifold_diagnostics"]


def test_shared_slot_store_curvature_sync():
    shared = _build_shared_geometry(dim=16, with_store=True)
    query = torch.randn(1, 2, 16)
    shared.run_shared_reasoning(
        query,
        content="curvature sync",
        mann_slot_index=3,
        ltm_slot_index=4,
    )
    store = shared.shared_slot_store
    assert float(store.slot_curvature[3].item()) != 0.0
    assert int(store.slot_geometry_code[3].item()) >= 0
    assert store.metadata[3]["shared_slot_geometry"] is True


def test_cross_memory_distance_with_topology_warp():
    shared = _build_shared_geometry(dim=16, with_topology=True)
    query = torch.randn(2, 16)
    diag = shared.compute_cross_memory_distance(
        query,
        mann_slot_index=1,
        ltm_slot_index=2,
        mann_geometry="torus",
        ltm_geometry="hyperbolic",
    )
    assert "distance_warped" in diag
    assert diag["distance_warped"] >= 0.0


def test_topology_v3_manifold_warp_blend_runs():
    topo = TopologyManagerV3(subsystems=("mann_ltm",))
    dist = torch.rand(2, 4)
    idx = torch.randint(0, 8, (2, 4))
    curv = torch.randn(8)
    q = torch.randn(2, 16)
    keys = torch.randn(8, 16)
    out = topo.warp_and_blend(
        dist,
        idx,
        curv,
        subsystem="mann_ltm",
        query_feat=q,
        key_feat=keys,
        alpha_torus=1.0,
    )
    assert out.shape == dist.shape
    assert torch.isfinite(out).all()


def test_manifold_retraction_helpers():
    x = torch.randn(4, 8)
    assert torch.isfinite(retract_sphere(x)).all()
    assert torch.isfinite(retract_poincare(x * 2.0)).all()
    assert torch.isfinite(retract_torus(x)).all()
    th = torus_add(torch.zeros(3), torch.tensor([3.5, -1.0, 0.5]))
    assert torch.all(th <= torch.pi)
    z1_re, z1_im = torch.randn(2, 4), torch.randn(2, 4)
    z2_re, z2_im = torch.randn(2, 4), torch.randn(2, 4)
    d = fubini_study_distance(z1_re, z1_im, z2_re, z2_im)
    assert torch.isfinite(d).all()
    ball = poincare_project_ball(x, max_norm=0.9)
    assert float(ball.norm(dim=-1).max().item()) <= 0.9 + 1e-5
