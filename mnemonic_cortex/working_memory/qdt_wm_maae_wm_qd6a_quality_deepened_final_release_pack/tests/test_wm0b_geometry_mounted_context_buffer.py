import torch

from mnemonic_cortex.working_memory import GeometryMountedContextBuffer, ContextTripletProjector, ContextDepthAdapter


def test_geometry_mounted_context_buffer_preserves_backward_api():
    buffer = GeometryMountedContextBuffer(dim=32, num_depths=8)
    ctx = torch.randn(2, 4, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    mounted, selected = buffer.mount(ctx, depth_state, requested_map="quantum_holographic")
    assert mounted.shape == depth_state.shape
    assert selected.name == "quantum_holographic"
    assert hasattr(selected, "trace")
    assert selected.trace.selected_map == "quantum_holographic"


def test_context_triplet_projector_shape():
    projector = ContextTripletProjector(dim=32)
    ctx = torch.randn(2, 4, 32)
    triplet = projector(ctx)
    assert triplet.shape == (2, 4, 3, 32)


def test_context_depth_adapter_shape():
    projector = ContextTripletProjector(dim=32)
    adapter = ContextDepthAdapter(dim=32, num_depths=8)
    ctx = torch.randn(2, 4, 32)
    triplet = projector(ctx)
    depth_weights = torch.ones(8)
    triplet_bias = torch.tensor([1.0, 0.5, 0.25])
    out = adapter(triplet, depth_weights, triplet_bias)
    assert out.shape == (2, 8, 1, 3, 32)
