import torch

from mnemonic_cortex.cps import ConsolidatedParamStore, UnifiedParamCfg
from mnemonic_cortex.cps_fuser import CPSFuser, FuserCfg
from mnemonic_cortex.memory_cgmn import EnhancedCGMNMemory
from mnemonic_cortex.memory_hg import EnhancedHyperGeometricMemory
from mnemonic_cortex.multi_cps import MultiCPSManager


def _mk_store_and_fuser():
    store = ConsolidatedParamStore(
        UnifiedParamCfg(d_euclid=16, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4)
    )
    fuser = CPSFuser(FuserCfg(d_model=16, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
    return store, fuser


def test_multi_cps_routes_prefixed_and_fallback_domains():
    mgr = MultiCPSManager()
    core, core_fuser = _mk_store_and_fuser()
    science, science_fuser = _mk_store_and_fuser()
    mgr.register("core", core, core_fuser).register("science", science, science_fuser)

    assert mgr.route("science:atp") == "science"
    assert mgr.route("unknown:item") == "core"
    assert mgr.route("key_without_prefix") == "core"
    assert set(mgr.domains()) == {"core", "science"}


def test_hg_flushes_pending_updates_before_read():
    torch.manual_seed(0)
    mem = EnhancedHyperGeometricMemory(input_dim=16, mem_slots=32, topk=4, holo_dim=64)
    mem.flush_every = 8  # keep write pending after one write call
    x = torch.randn(2, 3, 16)

    with torch.no_grad():
        mem(x, operation="write")
        assert len(mem._upd_idx_buffer) == 1
        before = mem.holograms_fft.detach().clone()
        _ = mem(x, operation="read")
        after = mem.holograms_fft.detach().clone()

    assert len(mem._upd_idx_buffer) == 0
    assert not torch.allclose(before, after)


def test_cgmn_write_only_updates_touched_slots():
    torch.manual_seed(0)
    mem = EnhancedCGMNMemory(input_dim=8, manifold_dim=4, mem_slots=8, slot_dim=6, topk=2)
    with torch.no_grad():
        baseline = mem.memory_slots.detach().clone()
        encoded = torch.randn(1, 1, mem.H)
        idx = torch.tensor([[[2, 2]]], dtype=torch.long)
        w = torch.tensor([[[1.0, 0.0]]], dtype=encoded.dtype)
        mem._write(encoded, (w, idx), ema=0.5)

        changed_mask = (mem.memory_slots - baseline).abs().sum(dim=-1) > 1e-8

    assert int(changed_mask.sum().item()) == 1
    assert bool(changed_mask[2].item())
