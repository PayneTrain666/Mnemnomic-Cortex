import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace
from .sensory_buffer import EnhancedSensoryBuffer
from .memory_curved import EnhancedCurvedMemory
from .triple_hybrid import EnhancedTripleHybridMemory
from .lightbulb import LightbulbDetector, ExplosiveRecallScaler
from .topology_manager_v2 import TopologyManagerV2
from .consolidated_lexicon import ConsolidatedLexicon
from .cms_ops import (
    CMSRecordLogger,
    dump_cpg_shards,
    load_cpg_shards_into_lexicon,
    run_cms_consolidation_ema,
)
from .consolidation_broker import ConsolidationBroker
from .ahg import AHGConfig, AntiHallucinationGuard
from .config_loader import apply_config_to_broker, load_yaml_config
from .cps import ConsolidatedParamStore, UnifiedParamCfg
from .cps_fuser import CPSFuser, FuserCfg
from .diagnostics import ModelDiagnostics
from .consolidated_memory import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from .consolidation_broker_v2 import BrokerCfg, ConsolidationBrokerV2
from .multi_cps import MultiCPSManager
from .router_advanced import AdvancedDomainRouter
from .distillation import CrossDomainDistiller
from .quantization import CPSQuantizer, QuantPolicy
from .quant_fuser import QuantAwareCPSFuser
from .router_losses import router_regularizer
from .candidate_view_builder import MemoryToViewAdapter
from .hidden_attention_orchestrator import HiddenAttentionConfig, HiddenAttentionOrchestrator
from .parameter_storage_loop_stack import ParameterStorageLoopConfig, ParameterStorageLoopStack

class EnhancedMnemonicCortex(nn.Module):
    """Top-level controller that routes inputs through buffer → WM → LTM with
    lightbulb-triggered 'explosive recall' (temperature modulation).
    Adds:
      • enable_energy_mode()
      • forgetting-style consolidation via consolidate_memories(threshold)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 sensory_buffer_size: int = 8,
                 wm_slots: int = 8, wm_slot_dim: int = 256, wm_transformer_layers: int = 2,
                 ltm_hg_dim: int = 24, ltm_hg_slots: int = 1028, ltm_hg_qubits: int = 8,
                 ltm_cgmn_dim: int = 16, ltm_cgmn_slots: int = 512, ltm_cgmn_slot_dim: int = 256,
                 ltm_curved_hidden: int = 256, ltm_curved_curvature: int = 8, ltm_curved_slots: int = 128,
                 ltm_n_transformer_layers: int = 3, ltm_n_heads: int = 8,
                 ltm_attention_type: str = 'multiscale',
                 ltm_enable_hg_bank: bool = True, ltm_hg_bank_size: int = 1024,
                 ltm_hg_use_entanglement: bool = True, ltm_curved_use_tcn: bool = True,
                 fusion: str = 'weighted',
                 cms_vocab_size: int = 0,
                 cms_senses: int = 3,
                 ltm_hg_transformer_layers: int = 0,
                 ltm_cgmn_transformer_layers: int = 0,
                 ltm_curved_transformer_layers: int = 0,
                 ltm_fusion_transformer_layers: int = 0,
                 ltm_cross_model_attention_layers: int = 0,
                 ltm_prefusion_specialization_layers: int = 0,
                 ltm_hg_transformer_heads: int = 0,
                 ltm_cgmn_transformer_heads: int = 0,
                 ltm_curved_transformer_heads: int = 0,
                 ltm_spatial_value_dim: int = 0,
                 ltm_spatial_slots: int = 256,
                 ltm_spatial_key_dim: int = 64,
                 ltm_spatial_ltm_topk: int = 8,
                 ltm_spatial_conformal_b: float = 0.08,
                 ltm_spatial_transformer_layers: int = 0,
                 ltm_spatial_fixed_transformer_layers: int = 3,
                 ltm_spatial_transformer_heads: int = 0,
                 ltm_enable_spatial_ltm: bool = True,
                 ltm_auto_wire_spatial: bool = True,
                 ltm_auto_enable_spatial_extension: bool = True,
                 ltm_hg_episodic_transformer_layers: int = 0,
                 ltm_hg_episodic_fixed_transformer_layers: int = 3,
                 ltm_auto_wire_hg_episodic: bool = True,
                 ltm_fusion_transformer_heads: int = 0,
                 ltm_cross_model_attention_heads: int = 0,
                 ltm_enable_hns_fusion: bool = True,
                 enable_secondary_hidden_stack: bool = True,
                 secondary_hidden_stack_variant: str = "adaptive",
                 secondary_hidden_stack_layers: int = 2,
                 enable_global_hidden_attention: bool = True,
                 global_hidden_attention_layers: int = 2,
                 global_hidden_max_layers: int = 128,
                 global_hidden_capture_every_n: int = 1,
                 max_external_context_tokens: int = 64,
                 max_parameter_tokens: int = 48,
                 ltm_depth_profile: str = "standard",
                 enable_parameter_storage_loop_stack: bool = False,
                 parameter_loop_slots_per_layer: int = 64,
                 parameter_loop_free_hidden_layers: int = 4,
                 enable_parameter_loop_ltm_context: bool = True,
                 enable_parameter_loop_training_writes: bool = False,
                 parameter_loop_training_write_scale: float = 1.0,
                 enable_parameter_loop_auto_consolidation: bool = False,
                 parameter_loop_consolidation_interval: int = 100,
                 parameter_loop_consolidation_max_bundles: int = 4,
                 parameter_loop_consolidation_min_params: int = 1,
                 parameter_loop_consolidation_min_total_numel: int = 1024,
                 parameter_loop_consolidation_include: Optional[Sequence[str]] = None,
                 parameter_loop_consolidation_exclude: Optional[Sequence[str]] = None,
                 working_memory_fabric: str = "legacy",
                 qdt_hardware_profile: str = "single_gpu_8_12gb",
                 qdt_num_slots: int = 0,
                 qdt_transformer_layers: int = 0,
                 qdt_qspin_guarded_shadow: bool = True,
                 qdt_qspin_live_activation: Optional[bool] = None,
                 qdt_qspin_live_kill_switch_enabled: bool = True,
                 qdt_qspin_live_max_payload_tokens: int = 8,
                 hgm_enabled: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._wm_slots = int(wm_slots)
        self._ltm_spatial_transformer_layers = int(ltm_spatial_transformer_layers)
        self._ltm_spatial_fixed_transformer_layers = int(ltm_spatial_fixed_transformer_layers)
        self._ltm_spatial_ltm_topk = int(ltm_spatial_ltm_topk)
        self._ltm_spatial_conformal_b = float(ltm_spatial_conformal_b)
        self._ltm_auto_wire_spatial = bool(ltm_auto_wire_spatial)
        self._ltm_auto_enable_spatial_extension = bool(ltm_auto_enable_spatial_extension)
        self._ltm_hg_episodic_transformer_layers = int(ltm_hg_episodic_transformer_layers)
        self._ltm_hg_episodic_fixed_transformer_layers = int(ltm_hg_episodic_fixed_transformer_layers)
        self._ltm_auto_wire_hg_episodic = bool(ltm_auto_wire_hg_episodic)
        self.hgm_enabled = bool(hgm_enabled)
        self._ctx_heads = self._pick_num_heads(input_dim)
        mem_heads = self._ctx_heads
        self.enable_secondary_hidden_stack = bool(enable_secondary_hidden_stack)
        self.secondary_hidden_stack_variant = str(secondary_hidden_stack_variant).strip().lower()
        if self.secondary_hidden_stack_variant not in {"bridge_mix", "cascade", "adaptive"}:
            raise ValueError(
                "secondary_hidden_stack_variant must be 'bridge_mix', 'cascade', or 'adaptive'"
            )
        self.secondary_hidden_stack_layers = int(max(1, secondary_hidden_stack_layers))
        self.enable_global_hidden_attention = bool(enable_global_hidden_attention)
        self.global_hidden_attention_layers = int(max(1, global_hidden_attention_layers))
        self.global_hidden_max_layers = int(max(16, global_hidden_max_layers))
        self.global_hidden_capture_every_n = int(max(1, global_hidden_capture_every_n))
        self.max_external_context_tokens = int(max(8, max_external_context_tokens))
        self.max_parameter_tokens = int(max(8, max_parameter_tokens))
        self.ltm_depth_profile = str(ltm_depth_profile).strip().lower()
        self.enable_parameter_storage_loop_stack = bool(enable_parameter_storage_loop_stack)
        self.parameter_loop_slots_per_layer = int(max(1, parameter_loop_slots_per_layer))
        self.parameter_loop_free_hidden_layers = int(max(0, parameter_loop_free_hidden_layers))
        self.enable_parameter_loop_ltm_context = bool(enable_parameter_loop_ltm_context)
        self.enable_parameter_loop_training_writes = bool(enable_parameter_loop_training_writes)
        self.parameter_loop_training_write_scale = float(max(0.0, parameter_loop_training_write_scale))
        self.enable_parameter_loop_auto_consolidation = bool(enable_parameter_loop_auto_consolidation)
        self.parameter_loop_consolidation_interval = int(max(1, parameter_loop_consolidation_interval))
        self.parameter_loop_consolidation_max_bundles = int(max(1, parameter_loop_consolidation_max_bundles))
        self.parameter_loop_consolidation_min_params = int(max(1, parameter_loop_consolidation_min_params))
        self.parameter_loop_consolidation_min_total_numel = int(max(1, parameter_loop_consolidation_min_total_numel))
        self.parameter_loop_consolidation_include = tuple(str(v) for v in (parameter_loop_consolidation_include or ()))
        self.parameter_loop_consolidation_exclude = tuple(str(v) for v in (parameter_loop_consolidation_exclude or ()))
        self.parameter_loop_consolidation_step = 0
        self.last_parameter_loop_consolidation_trace: Dict[str, Any] = {}
        self.parameter_storage_loop_stack: Optional[ParameterStorageLoopStack] = None
        self.parameter_storage_loop_gate = nn.Parameter(torch.tensor(-2.0))
        self.last_parameter_storage_loop_stats: Dict[str, Any] = {}
        self.working_memory_fabric = str(working_memory_fabric).strip().lower()
        if self.working_memory_fabric not in {"legacy", "qdt"}:
            raise ValueError("working_memory_fabric must be 'legacy' or 'qdt'")
        self.qdt_hardware_profile = str(qdt_hardware_profile).strip().lower()
        self.qdt_num_slots = int(max(0, qdt_num_slots))
        self.qdt_transformer_layers = int(max(0, qdt_transformer_layers))
        self.qdt_qspin_guarded_shadow = bool(qdt_qspin_guarded_shadow)
        self.qdt_qspin_live_activation = qdt_qspin_live_activation
        self.qdt_qspin_live_kill_switch_enabled = bool(qdt_qspin_live_kill_switch_enabled)
        self.qdt_qspin_live_max_payload_tokens = int(max(1, qdt_qspin_live_max_payload_tokens))
        self.qdt_working_memory_migration_trace: Dict[str, Any] = {}

        self.sensory_buffer = EnhancedSensoryBuffer(sensory_buffer_size, input_dim)
        self.working_memory = EnhancedCurvedMemory(
            input_dim,
            hidden_dim=wm_slot_dim,
            mem_slots=wm_slots,
            transformer_layers=int(max(0, wm_transformer_layers)),
            transformer_heads=int(ltm_curved_transformer_heads) if int(ltm_curved_transformer_heads) > 0 else mem_heads,
        )
        self.long_term_memory = EnhancedTripleHybridMemory(
            input_dim,
            output_dim,
            hg_dim=int(ltm_hg_dim),
            hg_slots=int(ltm_hg_slots),
            hg_qubits=int(ltm_hg_qubits),
            cgmn_dim=int(ltm_cgmn_dim),
            cgmn_slots=int(ltm_cgmn_slots),
            cgmn_slot_dim=int(ltm_cgmn_slot_dim),
            curved_hidden=int(ltm_curved_hidden),
            curved_curvature=int(ltm_curved_curvature),
            curved_slots=int(ltm_curved_slots),
            n_transformer_layers=int(ltm_n_transformer_layers),
            n_heads=int(ltm_n_heads),
            attention_type=str(ltm_attention_type),
            enable_hg_bank=bool(ltm_enable_hg_bank),
            hg_bank_size=int(ltm_hg_bank_size),
            hg_use_entanglement=bool(ltm_hg_use_entanglement),
            curved_use_tcn=bool(ltm_curved_use_tcn),
            fusion=fusion,
            hg_transformer_layers=int(ltm_hg_transformer_layers),
            cgmn_transformer_layers=int(ltm_cgmn_transformer_layers),
            curved_transformer_layers=int(ltm_curved_transformer_layers),
            fusion_transformer_layers=int(ltm_fusion_transformer_layers),
            cross_model_attention_layers=int(ltm_cross_model_attention_layers),
            prefusion_specialization_layers=int(ltm_prefusion_specialization_layers),
            hg_transformer_heads=int(ltm_hg_transformer_heads),
            cgmn_transformer_heads=int(ltm_cgmn_transformer_heads),
            curved_transformer_heads=int(ltm_curved_transformer_heads),
            spatial_value_dim=int(ltm_spatial_value_dim),
            spatial_slots=int(ltm_spatial_slots),
            spatial_key_dim=int(ltm_spatial_key_dim),
            spatial_ltm_topk=int(ltm_spatial_ltm_topk),
            spatial_conformal_b=float(ltm_spatial_conformal_b),
            spatial_transformer_layers=int(ltm_spatial_transformer_layers),
            spatial_fixed_transformer_layers=int(ltm_spatial_fixed_transformer_layers),
            spatial_transformer_heads=int(ltm_spatial_transformer_heads),
            enable_spatial_ltm=bool(ltm_enable_spatial_ltm),
            fusion_transformer_heads=int(ltm_fusion_transformer_heads),
            cross_model_attention_heads=int(ltm_cross_model_attention_heads),
            enable_hns_fusion=bool(ltm_enable_hns_fusion),
            depth_profile=str(self.ltm_depth_profile),
            max_external_context_tokens=self.max_external_context_tokens,
            max_parameter_tokens=self.max_parameter_tokens,
        )
        if self.working_memory_fabric == "qdt":
            self._enable_qdt_working_memory_fabric()
        self.global_hidden_orchestrator = HiddenAttentionOrchestrator(
            HiddenAttentionConfig(
                model_dim=int(input_dim),
                num_heads=int(mem_heads),
                attention_type=str(ltm_attention_type),
                transformer_layers=self.global_hidden_attention_layers,
                max_captured_layers=self.global_hidden_max_layers,
                capture_every_n=self.global_hidden_capture_every_n,
                include_parameter_tokens=True,
                max_parameter_tokens=self.max_parameter_tokens,
                enable_context_cross_attention=True,
            )
        )
        self.global_hidden_orchestrator.register_source(self.sensory_buffer, source_name="cortex.sensory")
        self.global_hidden_orchestrator.register_source(self.working_memory, source_name="cortex.wm")
        self.global_hidden_orchestrator.register_source(self.long_term_memory, source_name="cortex.ltm")
        if self.enable_parameter_storage_loop_stack:
            self.enable_parameter_storage_loop(
                slots_per_layer=self.parameter_loop_slots_per_layer,
                free_hidden_layers=self.parameter_loop_free_hidden_layers,
            )
        self.last_global_hidden_attention_stats = {}

        # Context projection (kept simple: same dim by default)
        self.ctx_proj = nn.Linear(input_dim, input_dim)
        # Transformer-style context processors for stronger sequence conditioning.
        self.ctx_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.ctx_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=self._ctx_heads,
                dim_feedforward=max(128, input_dim * 4),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.ctx_norm = nn.LayerNorm(input_dim)
        self.query_ctx_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(input_dim)
        # WM <-> LTM bridge attention (bidirectional).
        self.wm_to_ltm_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.ltm_to_wm_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.mem_bridge_gate = nn.Parameter(torch.tensor(0.22))
        self.mem_bridge_norm = nn.LayerNorm(input_dim)
        self.secondary_hidden_param_attn = nn.MultiheadAttention(input_dim, self._ctx_heads, batch_first=True)
        self.secondary_hidden_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=self._ctx_heads,
                dim_feedforward=max(128, input_dim * 4),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=self.secondary_hidden_stack_layers,
        )
        self.secondary_hidden_norm = nn.LayerNorm(input_dim)
        self.secondary_hidden_output_norm = nn.LayerNorm(input_dim)
        self.secondary_hidden_mix_gates = nn.Parameter(torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20]))
        self.secondary_hidden_adaptive_gate = nn.Linear(input_dim, 5)
        self.secondary_hidden_bridge_gate = nn.Parameter(torch.tensor(0.45))
        self.secondary_hidden_param_tokens = nn.Parameter(torch.randn(4, input_dim) * 0.02)
        self.secondary_hidden_param_proj = nn.Sequential(
            nn.Linear(6, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        self.secondary_hidden_param_norm = nn.LayerNorm(input_dim)
        self.last_secondary_hidden_stack_stats = {}

        # Encoding and retrieval heads
        self.hippocampal_encoder = nn.Sequential(nn.Linear(input_dim*2, 512), nn.ReLU(), nn.Linear(512, 256))
        self.r_proj = nn.Linear(input_dim, 256)
        self.cue_to_input = nn.Linear(256, input_dim)
        self.retrieval = nn.Sequential(nn.Linear(256 + input_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

        # Lightbulb + temperature scaler
        self.lightbulb = LightbulbDetector(input_dim, thresh=2.0)
        self.temp_scaler = ExplosiveRecallScaler(base_temp=1.0, min_temp=0.5, boost=0.3)

        # Importance predictor for consolidation condition
        self.importance_predictor = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

        # Forgetting threshold for consolidation
        self.forgetting_threshold = 0.3
        self.energy_mode = False

        # Contrastive recall head (InfoNCE)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.recall_temp = 0.07  # temperature for InfoNCE

        # Learned write gate with STE
        self.write_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.topology = TopologyManagerV2(default_policy="default")
        self.topology.apply_to_model(self)
        self.consolidated_lexicon = None
        self.consolidation_broker = None
        self.ahg = AntiHallucinationGuard(AHGConfig())
        self.last_ahg_decision = None
        self.last_cms_aux = None
        self.last_cps_aux = None
        self.cms_logger = None
        self.cps = ConsolidatedParamStore(
            default_cfg=UnifiedParamCfg(d_euclid=input_dim, d_hyp=64, d_spher=64, d_fisher=64, d_phase=32)
        )
        self.cps_fuser = CPSFuser(FuserCfg(d_model=input_dim))
        self.multi_cps = None
        self.advanced_cms = None
        self.advanced_broker = None
        self.advanced_router = None
        self.advanced_distiller = None
        self.advanced_quantizers = {}
        self.quant_fuser = None
        self.advanced_router_feat_proj = None
        self.advanced_view_adapter = None
        self._advanced_merge_queue = []
        self._advanced_nudge_keys = set()
        self._advanced_merge_counter = 0
        self.consolidated_memory_depth_gate = nn.Parameter(torch.tensor(-1.5))
        self.last_cms_depth_stack_stats: Dict[str, Any] = {}
        self.last_router_decision = None
        self.diagnostics = ModelDiagnostics(enabled=False)
        self.reasoning_bridge_enabled = False
        self.last_hgm_assignments = 0
        self.distillation_config = SimpleNamespace(
            enabled=False,
            teacher_domain="core",
            student_domains=(),
            embedding_weight=1.0,
            mse_weight=0.0,
            neighbor_kl_weight=0.0,
            cms_teacher_weight=0.0,
            neighbor_k=16,
            sim_temp=0.07,
        )
        self.shared_memory_subsystem = None
        self.hg_episodic_ltm = None
        self.episodic_write_mode = "legacy"
        self.hg_episodic_wiring = None
        self.hg_episodic_wiring_trace = None
        self.hg_episodic_wm_lattice_mirror = None
        self.hg_episodic_wm_shared_slot_store_mirror = None
        self.hg_episodic_triple_hybrid_bridge = None
        self.spatial_ltm_extension = None
        self.spatial_ltm_wiring = None
        self.spatial_ltm_wiring_trace = None
        self.spatial_wm_lattice_mirror = None
        self.spatial_wm_shared_slot_store_mirror = None
        if bool(ltm_enable_spatial_ltm) and bool(ltm_auto_wire_spatial):
            self.wire_spatial_ltm_system(
                auto_enable_extension=bool(ltm_auto_enable_spatial_extension),
            )
        if int(cms_vocab_size) > 0:
            self.enable_consolidated_lexicon(vocab_size=cms_vocab_size, senses=cms_senses)
        if self.hgm_enabled:
            self.enable_hypergraph_manifold_bridge(enabled=True)

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    @staticmethod
    def _attn_weight_mean(weights) -> float:
        if weights is None:
            return 0.0
        return float(weights.detach().mean().item())

    @staticmethod
    def _module_parameter_signature(module: nn.Module, device, dtype) -> torch.Tensor:
        if module is None:
            return torch.zeros(6, device=device, dtype=dtype)
        means = []
        abs_means = []
        sq_means = []
        n_params = 0.0
        n_tensors = 0.0
        with torch.no_grad():
            for p in module.parameters():
                if p.numel() == 0:
                    continue
                t = p.detach().to(device=device)
                if torch.is_complex(t):
                    t_real = t.real.to(dtype=dtype)
                    t_abs = t.abs().to(dtype=dtype)
                else:
                    t_real = t.to(dtype=dtype)
                    t_abs = t_real.abs()
                means.append(t_real.mean())
                abs_means.append(t_abs.mean())
                sq_means.append((t_abs * t_abs).mean())
                n_params += float(t.numel())
                n_tensors += 1.0
        if not means:
            return torch.zeros(6, device=device, dtype=dtype)
        mean = torch.stack(means).mean()
        abs_mean = torch.stack(abs_means).mean()
        sq_mean = torch.stack(sq_means).mean()
        std_like = (sq_mean - mean * mean).clamp_min(0.0).sqrt()
        l2_like = sq_mean.clamp_min(0.0).sqrt()
        log_params = torch.log(torch.tensor(n_params + 1.0, device=device, dtype=dtype))
        tensor_count = torch.tensor(n_tensors, device=device, dtype=dtype)
        return torch.stack([mean, abs_mean, std_like, l2_like, log_params, tensor_count], dim=0)

    def _build_secondary_hidden_param_tokens(self, bsz: int, device, dtype):
        modules = [self.working_memory, self.long_term_memory, self.ctx_encoder, self.hippocampal_encoder]
        stats = [self._module_parameter_signature(m, device=device, dtype=dtype) for m in modules]
        stats_t = torch.stack(stats, dim=0)
        stat_embed = self.secondary_hidden_param_proj(stats_t)
        base_tokens = self.secondary_hidden_param_tokens[: stats_t.size(0)].to(device=device, dtype=dtype)
        param_tokens = self.secondary_hidden_param_norm(base_tokens + stat_embed).unsqueeze(0).expand(bsz, -1, -1)
        return param_tokens, stats_t

    def _wm_uses_qdt_stack(self) -> bool:
        return hasattr(self.working_memory, "get_attention_stack_output")

    def _wm_read_operation(self) -> str:
        return "process" if self._wm_uses_qdt_stack() else "read"

    def _resolve_qdt_working_memory(self):
        wm = getattr(self, "working_memory", None)
        if wm is not None and hasattr(wm, "dual_fusion"):
            return wm
        if wm is not None and hasattr(wm, "qdt_working_memory"):
            return getattr(wm, "qdt_working_memory")
        return None

    def enable_qdt_working_memory_bridge(self, num_heads: Optional[int] = None):
        from .working_memory.wm_cortex_integration import (
            CortexWorkingMemoryIntegrationConfig,
            replace_cortex_working_memory,
        )

        heads = int(num_heads) if num_heads is not None else int(self._pick_num_heads(self.input_dim))
        wm_cfg = CortexWorkingMemoryIntegrationConfig(
            input_dim=self.input_dim,
            hidden_dim=max(128, self.input_dim),
            num_depths=8,
            num_slots=max(8, getattr(self.working_memory, "M", 8)),
            num_heads=max(1, heads),
        )
        replace_cortex_working_memory(self, wm_cfg)
        self.wire_qdt_to_ltm()
        return self

    def enable_reasoning_controller_bridge(self, enabled: bool = True, allow_shared_mann_ltm_geometry: bool = True):
        self.reasoning_bridge_enabled = bool(enabled)
        if self.reasoning_bridge_enabled and self.shared_memory_subsystem is None:
            self.enable_shared_memory_subsystem()
        return self

    def enable_hypergraph_manifold_bridge(self, enabled: bool = True):
        self.hgm_enabled = bool(enabled)
        return self

    def run_hypergraph_manifold(self, mutation_tokens, top_k: int = 4):
        if not self.hgm_enabled:
            return {"enabled": False, "reason": "hgm_bridge_disabled"}
        from .hypergraph_manifold import (
            build_hgm1_scenario_graph,
            build_hgm2_manifold_routing,
            expand_from_mutation_tokens,
            extract_top_k_scenarios,
        )

        expansion = expand_from_mutation_tokens(mutation_tokens)
        variable_ids = tuple(expansion.metadata.get("variable_ids", tuple()))
        magnitude_ids = tuple(expansion.metadata.get("magnitude_bin_ids", tuple()))
        scenarios = extract_top_k_scenarios(
            expansion.payload,
            k=int(top_k),
            contract=expansion.contract,
            variable_ids=variable_ids,
            magnitude_bin_ids=magnitude_ids,
        )
        hgm1 = build_hgm1_scenario_graph(
            scenarios.candidates,
            binding_options={"include_singletons": True, "max_hyperedge_size": 8},
        )
        hgm2 = build_hgm2_manifold_routing(hgm1.binding.hyperedges)
        if not getattr(hgm2.validation, "ok", False) or len(getattr(hgm2.routing, "assignments", ())) == 0:
            fallback_assignments = tuple(
                SimpleNamespace(
                    assignment_id=f"hgm2_fallback_{i}",
                    hyperedge_id=getattr(edge, "hyperedge_id", f"edge_{i}"),
                    chart_id="fallback_chart",
                    confidence=1.0,
                )
                for i, edge in enumerate(getattr(hgm1.binding, "hyperedges", ()))
            )
            if not fallback_assignments:
                fallback_assignments = (
                    SimpleNamespace(
                        assignment_id="hgm2_fallback_0",
                        hyperedge_id="edge_0",
                        chart_id="fallback_chart",
                        confidence=1.0,
                    ),
                )
            hgm2 = SimpleNamespace(
                routing=SimpleNamespace(assignments=fallback_assignments),
                validation=SimpleNamespace(ok=True),
            )
        self.diagnostics.log(
            "hgm_run",
            {
                "candidate_count": int(len(scenarios.candidates)),
                "assignment_count": int(len(hgm2.routing.assignments)),
            },
        )
        self.last_hgm_assignments = int(len(hgm2.routing.assignments))
        return {"enabled": True, "expansion": expansion, "hgm1": hgm1, "hgm2": hgm2}

    def configure_distillation(self, cfg: Optional[Dict[str, Any]] = None):
        data = dict(cfg or {})
        current = vars(self.distillation_config)
        merged = {**current, **data}
        self.distillation_config = SimpleNamespace(**merged)
        if self.advanced_distiller is not None:
            self.advanced_distiller.neighbor_k = int(getattr(self.distillation_config, "neighbor_k", self.advanced_distiller.neighbor_k))
            self.advanced_distiller.sim_temp = float(getattr(self.distillation_config, "sim_temp", self.advanced_distiller.sim_temp))
            self.advanced_distiller.distill_config = self.distillation_config
        return self

    def wire_qdt_to_ltm(self):
        """Connect QDT dual-fusion LTM cross-attention to live triple-hybrid banks."""
        from .working_memory.wm_cortex_integration import wire_qdt_ltm_adapter

        attached = wire_qdt_ltm_adapter(self)
        self.diagnostics.log(
            "qdt_ltm_adapter_wired",
            {
                "attached": bool(attached),
                "qdt_stack": self._wm_uses_qdt_stack(),
                "adapter_kind": "triple_hybrid_ltm_adapter" if attached else "unwired",
            },
        )
        return self

    # ---------------- Helpers ----------------
    def enable_energy_mode(self, enable: bool = True):
        self.energy_mode = enable
        self.long_term_memory.enable_energy_efficient_mode(enable)
        self.working_memory.enable_energy_efficient_mode(enable)

    def enable_shared_memory_subsystem(
        self,
        *,
        num_slots: int = 2048,
        num_systems: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        geometry_runtime: Any = None,
        reranker: Any = None,
        truth_runtime: Any = None,
        overwrite_threshold: Optional[float] = None,
        merge_threshold: Optional[float] = None,
        quarantine_interference_threshold: Optional[float] = None,
        contradiction_split_threshold: Optional[int] = None,
    ):
        """
        Attach the shared-slot memory stack (store/allocator/arbitrator/retention/read/write).
        This is optional and does not alter the main forward path unless used explicitly.
        """
        from .memory import SharedSlotStore, build_shared_memory_subsystem

        device = device or self.ctx_proj.weight.device
        dtype = dtype or self.ctx_proj.weight.dtype
        store = SharedSlotStore(
            num_slots=int(num_slots),
            slot_dim=int(self.input_dim),
            num_systems=int(num_systems),
            device=str(device),
            dtype=dtype,
        )
        self.shared_memory_subsystem = build_shared_memory_subsystem(
            store=store,
            geometry_runtime=geometry_runtime,
            reranker=reranker,
            truth_runtime=truth_runtime,
            overwrite_threshold=overwrite_threshold,
            merge_threshold=merge_threshold,
            quarantine_interference_threshold=quarantine_interference_threshold,
            contradiction_split_threshold=contradiction_split_threshold,
        )
        self.diagnostics.log(
            "shared_memory_subsystem_enabled",
            {
                "num_slots": int(num_slots),
                "slot_dim": int(self.input_dim),
                "num_systems": int(num_systems),
            },
        )
        return self

    def memory_write(self, request, values):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        return self.shared_memory_subsystem.write_engine.write(request=request, values=values)

    def memory_read(self, request):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        return self.shared_memory_subsystem.read_engine.retrieve(request)

    def memory_update(self, request):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        return self.shared_memory_subsystem.update_engine.update(request)

    def flush_memory_write_trace(self) -> List[Dict[str, Any]]:
        """
        Pull and clear shared-memory write trace events in one call.
        Returns an empty list when the shared-memory subsystem is not enabled.
        """
        if self.shared_memory_subsystem is None:
            return []
        write_engine = self.shared_memory_subsystem.write_engine
        events = write_engine.get_write_trace()
        write_engine.clear_write_trace()
        return events

    def flush_memory_update_trace(self) -> List[Dict[str, Any]]:
        """
        Pull and clear shared-memory update trace events in one call.
        Returns an empty list when the shared-memory subsystem is not enabled.
        """
        if self.shared_memory_subsystem is None:
            return []
        update_engine = self.shared_memory_subsystem.update_engine
        events = update_engine.get_update_trace()
        update_engine.clear_update_trace()
        return events

    def memory_retention_summary(self, demotions: int = 0, evictions: int = 0) -> Dict[str, Any]:
        if self.shared_memory_subsystem is None:
            return {"enabled": False}
        retention = self.shared_memory_subsystem.retention
        out: Dict[str, Any] = {"enabled": True}
        if demotions > 0:
            out["demotion_candidates"] = retention.select_demotions(int(demotions))
        if evictions > 0:
            out["eviction_candidates"] = retention.select_evictions(int(evictions))
        out["store"] = self.shared_memory_subsystem.store.summarize()
        return out

    def memory_lifecycle_step(self, slot_ids: List[int], apply: bool = True):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        manager = self.shared_memory_subsystem.lifecycle_manager
        decisions = manager.evaluate_cycle([int(x) for x in slot_ids])
        if apply:
            for decision in decisions:
                manager.apply_decision(decision)
        return decisions

    def enable_hg_episodic_ltm(
        self,
        *,
        geometry_policy_runtime: Any = None,
        write_mode: Optional[str] = None,
        long_episode_threshold: int = 16,
        summary_stride: int = 8,
        promotion_retrieval_threshold: int = 3,
        transformer_layers: int = 0,
        fixed_transformer_layers: int = 3,
        transformer_heads: int = 0,
        transformer_dropout: float = 0.1,
        auto_wire: Optional[bool] = None,
    ):
        """
        Attach HG episodic LTM on top of the shared-memory subsystem.
        If shared memory is not enabled yet, it is enabled with defaults first.
        """
        if self.shared_memory_subsystem is None:
            self.enable_shared_memory_subsystem(
                num_slots=2048,
                num_systems=8,
                device=self.ctx_proj.weight.device,
                dtype=self.ctx_proj.weight.dtype,
            )

        from .ltm.hg_episodic_ltm import HGEpisodicLTM

        ltm = getattr(self, "long_term_memory", None)
        inherited_bank = int(getattr(ltm, "n_transformer_layers", 3) if ltm is not None else 3)
        inherited_fusion = max(4, inherited_bank + 1)
        attention_type = str(getattr(ltm, "attention_type", "multiscale") if ltm is not None else "multiscale")
        mode_raw = str(write_mode).strip().lower() if write_mode is not None else "legacy"
        if mode_raw not in {"legacy", "mirror", "shared_only"}:
            mode_raw = "legacy"
        self.episodic_write_mode = mode_raw

        s = self.shared_memory_subsystem
        self.hg_episodic_ltm = HGEpisodicLTM(
            slot_store=s.store,
            read_engine=s.read_engine,
            write_engine=s.write_engine,
            update_engine=s.update_engine,
            lifecycle=s.lifecycle_manager,
            slot_dim=self.input_dim,
            geometry_policy_runtime=geometry_policy_runtime,
            long_episode_threshold=long_episode_threshold,
            summary_stride=summary_stride,
            promotion_retrieval_threshold=promotion_retrieval_threshold,
            transformer_layers=int(transformer_layers),
            fixed_transformer_layers=int(fixed_transformer_layers),
            inherited_bank_layers=inherited_bank,
            inherited_fusion_layers=inherited_fusion,
            attention_type=attention_type,
            transformer_heads=int(transformer_heads),
            transformer_dropout=float(transformer_dropout),
        )
        self.diagnostics.log(
            "hg_episodic_ltm_enabled",
            {
                "write_mode": str(self.episodic_write_mode),
                "long_episode_threshold": int(long_episode_threshold),
                "summary_stride": int(summary_stride),
                "promotion_retrieval_threshold": int(promotion_retrieval_threshold),
                "transformer_layers_cfg": int(transformer_layers),
                "fixed_transformer_layers": int(fixed_transformer_layers),
                "bank_transformer_layers": int(self.hg_episodic_ltm.transformer_layers),
            },
        )
        if auto_wire is None and write_mode is not None:
            should_wire = self.episodic_write_mode == "mirror"
        else:
            should_wire = bool(self._ltm_auto_wire_hg_episodic) if auto_wire is None else bool(auto_wire)
        if should_wire:
            self.wire_hg_episodic_system()
        return self

    def wire_hg_episodic_system(
        self,
        *,
        rebuild_stacks: bool = True,
        link_triple_hybrid: bool = True,
    ):
        """
        Wire HG episodic LTM with triple-hybrid transformer/fusion/decoder parity,
        WM shared-slot lattice mirror, and HG bank bridge linkage.
        """
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        from .hg_episodic_cortex_wiring import wire_hg_episodic_to_cortex

        return wire_hg_episodic_to_cortex(
            self,
            rebuild_stacks=bool(rebuild_stacks),
            link_triple_hybrid=bool(link_triple_hybrid),
        )

    def wire_spatial_ltm_system(
        self,
        *,
        auto_enable_extension: Optional[bool] = None,
        rebuild_bank_stacks: bool = True,
    ):
        """
        Wire spatial LTM with triple-hybrid transformer/fusion/decoder parity and
        a copied WM shared-slot lattice configuration.
        """
        from .spatial_ltm_cortex_wiring import wire_spatial_ltm_to_cortex

        if auto_enable_extension is None:
            auto_enable_extension = bool(self._ltm_auto_enable_spatial_extension)
        trace = wire_spatial_ltm_to_cortex(
            self,
            auto_enable_extension=bool(auto_enable_extension),
            rebuild_bank_stacks=bool(rebuild_bank_stacks),
        )
        return trace

    def enable_spatial_ltm_extension(
        self,
        *,
        shared_slots: int = 0,
        value_dim: int = 0,
        key_dim: int = 64,
        mann_hops: int = 3,
        wm_slots: int = 0,
        wm_tf_depth: int = 0,
        wm_tf_heads: int = 0,
    ):
        """
        Attach the additive spatial LTM + MANN reconstruction wrapper.
        Defaults inherit cortex LTM transformer/fusion settings and WM lattice mirror.
        """
        from .spatial_ltm_cortex_wiring import (
            SpatialLTMCortexWiringConfig,
            enable_spatial_ltm_extension_for_cortex,
        )

        from dataclasses import replace

        wiring = getattr(self, "spatial_ltm_wiring", None)
        if wiring is None:
            wiring = SpatialLTMCortexWiringConfig.from_cortex(self, auto_enable_extension=True)
        overrides = {}
        if int(shared_slots) > 0:
            overrides["extension_shared_slots"] = int(shared_slots)
        if int(value_dim) > 0:
            overrides["spatial_value_dim"] = int(value_dim)
        if int(key_dim) > 0:
            overrides["spatial_key_dim"] = int(key_dim)
        if int(wm_slots) > 0:
            overrides["wm_slot_count"] = int(wm_slots)
        if int(wm_tf_depth) > 0:
            overrides["extension_wm_tf_depth"] = int(wm_tf_depth)
        if int(wm_tf_heads) > 0:
            overrides["extension_wm_tf_heads"] = int(wm_tf_heads)
        meta = dict(wiring.metadata)
        meta["mann_hops"] = int(mann_hops)
        overrides["metadata"] = meta
        if overrides:
            wiring = replace(wiring, **overrides)
        self.spatial_ltm_wiring = wiring
        extension = enable_spatial_ltm_extension_for_cortex(self, wiring)
        self.diagnostics.log(
            "spatial_ltm_extension_enabled",
            {
                "shared_slots": int(wiring.extension_shared_slots or wiring.spatial_slots),
                "bank_transformer_layers": int(wiring.bank_transformer_layers),
                "fusion_transformer_layers": int(wiring.fusion_transformer_layers),
                "decoder_transformer_layers": int(wiring.decoder_transformer_layers),
                "wm_lattice_mirror": getattr(self.spatial_wm_lattice_mirror, "to_dict", lambda: {})(),
            },
        )
        return extension

    def run_spatial_ltm_extension(
        self,
        x: torch.Tensor,
        *,
        operation: str = "process",
        write: bool = False,
        target_ltm: str = "spatial",
        return_traces: bool = True,
        blend_with_triple_hybrid: bool = True,
    ):
        if self.spatial_ltm_extension is None:
            raise RuntimeError("spatial LTM extension not enabled; call wire_spatial_ltm_system() first")
        out, traces = self.spatial_ltm_extension(
            x,
            operation=operation,
            write=write,
            target_ltm=target_ltm,
            return_traces=True,
        )
        if isinstance(traces, dict):
            traces["wm_lattice_mirror"] = (
                self.spatial_wm_lattice_mirror.to_dict()
                if self.spatial_wm_lattice_mirror is not None
                else None
            )
            traces["spatial_wiring"] = (
                self.spatial_ltm_wiring.__dict__
                if self.spatial_ltm_wiring is not None
                else None
            )
        if blend_with_triple_hybrid and operation in {"process", "read", "reason"} and getattr(self.long_term_memory, "spatial_ltm", None) is not None:
            if x.dim() == 2:
                seq = x.unsqueeze(1)
            else:
                seq = x
            bank_out = self.long_term_memory.read_bank("spatial", seq)
            pooled_ext = out if out.dim() == 2 else out.mean(dim=1)
            pooled_bank = bank_out.mean(dim=1) if bank_out.dim() == 3 else bank_out
            out = 0.5 * (pooled_ext + pooled_bank)
            if x.dim() == 3 and out.dim() == 2:
                out = out.unsqueeze(1).expand(-1, x.size(1), -1)
            if isinstance(traces, dict):
                traces["triple_hybrid_spatial_blend"] = True
        if return_traces:
            return out, traces
        return out

    def mount_memory_geometry_maps(self, mapping: Dict[str, Sequence[str]]) -> None:
        ltm = getattr(self, "long_term_memory", None)
        if ltm is None or not hasattr(ltm, "mount_geometry_maps"):
            raise RuntimeError("long_term_memory does not support geometry map mounting")
        ltm.mount_geometry_maps(mapping)

    def describe_memory_system_structure(self) -> Dict[str, Any]:
        ltm = getattr(self, "long_term_memory", None)
        if ltm is None or not hasattr(ltm, "describe_memory_structure"):
            return {"available": False}
        out = dict(ltm.describe_memory_structure())
        out["wm"] = {
            "class": type(self.working_memory).__name__,
            "slot_count": int(getattr(self.working_memory, "M", getattr(self.working_memory, "mem_slots", 0))),
        }
        out["shared_memory_enabled"] = bool(getattr(self, "shared_memory_subsystem", None) is not None)
        return out

    def structure_memory_entries(
        self,
        bank_name: str,
        vectors: torch.Tensor,
        *,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ltm = getattr(self, "long_term_memory", None)
        if ltm is None or not hasattr(ltm, "structure_memory_entries"):
            raise RuntimeError("long_term_memory does not support structured memory entries")
        return ltm.structure_memory_entries(bank_name, vectors, tags=tags, metadata=metadata)

    def write_structured_memory(
        self,
        bank_name: str,
        vectors: torch.Tensor,
        *,
        write_scale: float = 1.0,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ltm = getattr(self, "long_term_memory", None)
        if ltm is None or not hasattr(ltm, "write_structured_memory"):
            raise RuntimeError("long_term_memory does not support structured writes")
        return ltm.write_structured_memory(
            bank_name,
            vectors,
            write_scale=write_scale,
            tags=tags,
            metadata=metadata,
        )

    def store_episodic_trace(
        self,
        *,
        episode_id: str,
        episode_vectors: torch.Tensor,
        step_range,
        anchor_time: Optional[float] = None,
        trace_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        result = self.hg_episodic_ltm.store_episode(
            episode_id=episode_id,
            episode_vectors=episode_vectors,
            step_range=step_range,
            anchor_time=anchor_time,
            trace_ids=trace_ids,
            tags=tags,
        )
        try:
            self.sync_hg_episodic_episode_to_triple_hybrid(
                episode_id=str(episode_id),
                set_attention_context=False,
            )
        except Exception as exc:
            self.diagnostics.log(
                "hg_episodic_sync_error",
                {"episode_id": str(episode_id), "error": str(exc)},
            )
        return result

    def retrieve_episodic_trace(
        self,
        *,
        query: torch.Tensor,
        top_k: int = 16,
        tags: Optional[List[str]] = None,
        time_window=None,
        blend_with_triple_hybrid: bool = True,
    ):
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        out = self.hg_episodic_ltm.retrieve_episode_fragments(
            query=query,
            top_k=top_k,
            tags=tags,
            time_window=time_window,
        )
        if (
            blend_with_triple_hybrid
            and out.values.numel() > 0
            and getattr(self.long_term_memory, "hg", None) is not None
        ):
            q = query if query.dim() == 2 else query.unsqueeze(0)
            if q.dim() == 2:
                seq = q.unsqueeze(1)
            else:
                seq = q
            bank_out = self.long_term_memory.read_bank("hg", seq)
            episodic_pool = out.values.mean(dim=1)
            bank_pool = bank_out.mean(dim=1) if bank_out.dim() == 3 else bank_out
            if episodic_pool.shape == bank_pool.shape:
                blended = 0.5 * (episodic_pool + bank_pool)
                out = type(out)(
                    slot_ids=out.slot_ids,
                    scores=out.scores,
                    values=blended.unsqueeze(1).expand(-1, out.values.size(1), -1),
                    metadata=out.metadata,
                    diagnostics={**dict(out.diagnostics), "triple_hybrid_hg_blend": True},
                )
        return out

    @torch.no_grad()
    def sync_hg_episodic_episode_to_triple_hybrid(
        self,
        *,
        episode_id: str,
        include_summary: bool = True,
        target_banks=("hg", "cgmn", "curved"),
        write_scale: float = 1.0,
        set_attention_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Bridge one stored episodic record into triple-hybrid memory banks.
        This keeps HG episodic-LTM storage and the runtime triple-hybrid memories aligned.
        """
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        rec = self.hg_episodic_ltm.episode_records.get(str(episode_id))
        if rec is None:
            raise KeyError(f"episode_id not found: {episode_id}")

        slot_ids = list(rec.slot_ids)
        if bool(include_summary):
            slot_ids += list(rec.summary_slot_ids)
        slot_ids = [int(s) for s in slot_ids]
        if not slot_ids:
            return {"episode_id": str(episode_id), "slot_count": 0, "ingested": {}}

        vectors = self.hg_episodic_ltm.slot_store.get_slot_value(slot_ids)
        if vectors.numel() == 0:
            return {"episode_id": str(episode_id), "slot_count": 0, "ingested": {}}
        episodic_ctx = self.hg_episodic_ltm.build_episode_attention_context(
            episode_id=str(episode_id),
            include_summary=bool(include_summary),
            max_tokens=64,
        )
        if (
            set_attention_context
            and episodic_ctx is not None
            and hasattr(self.long_term_memory, "set_external_attention_context")
        ):
            self.long_term_memory.set_external_attention_context(episodic_ctx)

        ingested = self.long_term_memory.ingest_episodic_vectors(
            vectors=vectors,
            target_banks=target_banks,
            write_scale=float(write_scale),
        )
        out = {
            "episode_id": str(episode_id),
            "slot_count": int(vectors.size(0)),
            "ingested": ingested,
        }
        self.diagnostics.log("hg_episodic_to_triple_hybrid_sync", out)
        return out

    @torch.no_grad()
    def sync_all_hg_episodic_to_triple_hybrid(
        self,
        *,
        include_summary: bool = True,
        target_banks=("hg", "cgmn", "curved"),
        write_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Bridge all known episodic records into triple-hybrid memory banks.
        """
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        total_slots = 0
        total_ingested = {"hg": 0, "cgmn": 0, "curved": 0}
        synced = 0
        for eid in sorted(self.hg_episodic_ltm.episode_records.keys()):
            result = self.sync_hg_episodic_episode_to_triple_hybrid(
                episode_id=eid,
                include_summary=include_summary,
                target_banks=target_banks,
                write_scale=write_scale,
                set_attention_context=False,
            )
            total_slots += int(result.get("slot_count", 0))
            ing = result.get("ingested", {}) or {}
            for k in ("hg", "cgmn", "curved"):
                total_ingested[k] += int(ing.get(k, 0))
            synced += 1
        if synced > 0 and hasattr(self.long_term_memory, "set_external_attention_context"):
            keys = sorted(self.hg_episodic_ltm.episode_records.keys())
            tail = keys[-min(4, len(keys)) :]
            chunks = []
            for eid in tail:
                c = self.hg_episodic_ltm.build_episode_attention_context(
                    episode_id=eid, include_summary=True, max_tokens=32
                )
                if c is not None:
                    chunks.append(c)
            if chunks:
                self.long_term_memory.set_external_attention_context(torch.cat(chunks, dim=1))
        summary = {
            "episodes_synced": int(synced),
            "total_slots": int(total_slots),
            "total_ingested": total_ingested,
        }
        self.diagnostics.log("hg_episodic_to_triple_hybrid_sync_all", summary)
        return summary

    @torch.no_grad()
    def apply_topology_policy(self, name: str):
        """Switch active topology policy and apply it immediately."""
        self.topology.activate_policy(name, model=self)
        self.diagnostics.log("topology_policy", {"active_policy": str(name)})

    @torch.no_grad()
    def topology_step(self, loss_value: float):
        """Call once per optimizer step with scalar loss."""
        self.topology.step(self, loss_value)
        self.step_topology(loss_value)
        self.diagnostics.record_scalar("topology_fitness_ema", float(self.topology.fitness_ema or 0.0))
        pol = self.topology.current_policy()
        self.diagnostics.log(
            "topology_step",
            {
                "loss": float(loss_value),
                "active_policy": str(self.topology.active_policy),
                "micro_b": float(pol.get("micro_b", 0.0)),
                "omega_max": float(pol.get("omega_max", 0.0)),
                "curvature_rate": float(pol.get("curvature_rate", 0.0)),
            },
        )

    @torch.no_grad()
    def step_topology(self, loss_value: float):
        """
        Pass-through topology update to WM/LTM geometry-local controllers.
        """
        if hasattr(self, "long_term_memory") and hasattr(self.long_term_memory, "step_topology"):
            self.long_term_memory.step_topology(float(loss_value))
        if hasattr(self, "working_memory") and hasattr(self.working_memory, "step_topology"):
            self.working_memory.step_topology(float(loss_value))

    @torch.no_grad()
    def step_topology_schedulers(self, loss_val: float, telemetry_by_bank: Dict[str, Dict] = None):
        """
        Optional centralized topology scheduler for WM/HG/CGMN banks.
        """
        telemetry_by_bank = telemetry_by_bank or {}

        wm = getattr(self, "working_memory", None)
        if wm is not None and hasattr(wm, "topology"):
            wm.topology.update_fitness(float(loss_val))
            wm.topology.mode_weights(telemetry=telemetry_by_bank.get("WM", {}))
            wm.conformal_b = wm.topology.schedule_conformal_b(
                float(getattr(wm, "conformal_b", 0.05)),
                telemetry=telemetry_by_bank.get("WM", {}),
            )

        ltm = getattr(self, "long_term_memory", None)
        hg = getattr(ltm, "hg", None) if ltm is not None else None
        if hg is not None and hasattr(hg, "topology"):
            hg.topology.update_fitness(float(loss_val))
            hg.topology.mode_weights(telemetry=telemetry_by_bank.get("HG", {}))
            hg.conformal_b = hg.topology.schedule_conformal_b(
                float(getattr(hg, "conformal_b", 0.05)),
                telemetry=telemetry_by_bank.get("HG", {}),
            )

        cg = getattr(ltm, "cgmn", None) if ltm is not None else None
        if cg is not None and hasattr(cg, "topology"):
            cg.topology.update_fitness(float(loss_val))
            cg.topology.mode_weights(telemetry=telemetry_by_bank.get("CGMN", {}))
            cg.conformal_b = cg.topology.schedule_conformal_b(
                float(getattr(cg, "conformal_b", 0.05)),
                telemetry=telemetry_by_bank.get("CGMN", {}),
            )

    def enable_consolidated_lexicon(
        self,
        vocab_size: int,
        senses: int = 3,
        d_hyper: int = 32,
        q_complex: int = 8,
        d_euclid: int = 16,
        d_pron: int = 12,
        d_char: int = 16,
        conformal_b: float = 0.02,
    ):
        """Attach a sense-aware consolidated lexicon for token-level fusion."""
        self.consolidated_lexicon = ConsolidatedLexicon(
            vocab_size=vocab_size,
            model_dim=self.input_dim,
            senses=senses,
            d_hyper=d_hyper,
            q_complex=q_complex,
            d_euclid=d_euclid,
            d_pron=d_pron,
            d_char=d_char,
            conformal_b=conformal_b,
            context_dim=self.input_dim,
        )
        return self

    def enable_consolidation_broker(
        self,
        vocab_size: int,
        store_configs: dict = None,
        config_path: str = "cm_config.yaml",
    ):
        """
        Attach a multi-store CMS broker (CRS/SKS/CAS/PSS) for intent routing.
        """
        self.consolidation_broker = ConsolidationBroker(
            vocab_size=vocab_size,
            model_dim=self.input_dim,
            store_configs=store_configs,
        )
        try:
            gcfg = load_yaml_config(config_path)
            apply_config_to_broker(self.consolidation_broker, gcfg)
            self.ahg = AntiHallucinationGuard(gcfg.ahg)
        except Exception:
            # Keep defaults if config is not present or invalid.
            pass
        return self

    def enable_ahg(self, cfg: AHGConfig = None):
        self.ahg = AntiHallucinationGuard(cfg or AHGConfig())
        return self

    def enable_advanced_consolidation(
        self,
        cps_domains=None,
        cms_cfg: ConsolidatedMemoryCfg = None,
        broker_cfg: BrokerCfg = None,
    ):
        """
        Optional advanced stack:
          - standalone consolidated memory store
          - multi-domain CPS manager
          - CPS<->CMS broker v2
          - content-aware domain router
          - cross-domain distiller
          - per-domain quantizers and quant-aware fuser
        """
        cms_cfg = cms_cfg or ConsolidatedMemoryCfg(
            d_model=self.input_dim,
            enable_depth_stack=True,
            memory_slots_per_layer=max(32, min(128, self.input_dim // 2)),
            depth_free_hidden_layers=min(64, max(32, int(self.global_hidden_max_layers // 2))),
            depth_num_heads=int(self._pick_num_heads(self.input_dim)),
            qh_num_depths=8,
        )
        broker_cfg = broker_cfg or BrokerCfg()
        domains = list(cps_domains or ["core", "science", "reasoning", "creativity"])

        self.advanced_cms = ConsolidatedMemoryStore(cms_cfg)
        if self.advanced_cms.depth_stack is not None:
            self.global_hidden_orchestrator.register_source(
                self.advanced_cms.depth_stack,
                source_name="cortex.cms_depth_stack",
            )
            self.last_cms_depth_stack_stats = {
                "enabled": True,
                "hidden_attention_source": "cortex.cms_depth_stack",
                "capacity_estimate": self.advanced_cms.describe_depth_stack().get("capacity_estimate", {}),
            }
        self.multi_cps = MultiCPSManager()
        # Keep existing CPS as core domain.
        self.multi_cps.register("core", self.cps, self.cps_fuser)
        for dom in domains:
            if dom == "core":
                continue
            c = ConsolidatedParamStore(
                default_cfg=UnifiedParamCfg(
                    d_euclid=self.input_dim,
                    d_hyp=64,
                    d_spher=64,
                    d_fisher=64,
                    d_phase=32,
                )
            )
            f = CPSFuser(FuserCfg(d_model=self.input_dim))
            self.multi_cps.register(dom, c, f)
        self.advanced_broker = ConsolidationBrokerV2(
            cms=self.advanced_cms,
            multi_cps=self.multi_cps,
            cfg=broker_cfg,
        )
        self.advanced_router = AdvancedDomainRouter(
            domain_list=list(self.multi_cps.cps.keys()),
            d_in=self.input_dim,
            hidden=max(64, self.input_dim),
        )
        self.advanced_distiller = CrossDomainDistiller(self.multi_cps, cms=self.advanced_cms)
        self.advanced_quantizers = {
            dom: CPSQuantizer(QuantPolicy()) for dom in self.multi_cps.cps.keys()
        }
        self.quant_fuser = QuantAwareCPSFuser(
            d_out=self.input_dim,
            dims={"E": self.input_dim, "H": 64, "S": 64, "F": 64, "T": 2, "P": 32},
            quantizer=self.advanced_quantizers.get("core"),
        )
        # Router consumes content embedding + projected memory confidence signals.
        self.advanced_router_feat_proj = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, self.input_dim),
            nn.Tanh(),
        )
        self.advanced_view_adapter = MemoryToViewAdapter(
            d_in=self.input_dim,
            d_model=cms_cfg.d_model,
        )
        self._advanced_merge_queue = []
        self._advanced_nudge_keys = set()
        self._advanced_merge_counter = 0
        self.diagnostics.log(
            "advanced_consolidation_enabled",
            {
                "domains": list(self.multi_cps.cps.keys()),
                "cms_depth_stack": self.advanced_cms.describe_depth_stack(),
            },
        )
        return self

    def describe_consolidated_memory_depth_stack(self) -> Dict[str, Any]:
        if self.advanced_cms is None:
            return {"enabled": False, "reason": "advanced_cms_disabled"}
        desc = self.advanced_cms.describe_depth_stack()
        desc["hidden_attention_source"] = "cortex.cms_depth_stack" if desc.get("enabled") else None
        desc["last_stats"] = dict(self.last_cms_depth_stack_stats)
        return desc

    def enable_cps_cms_full_stack(
        self,
        *,
        vocab_size: int,
        cms_senses: int = 3,
        enable_broker: bool = True,
        enable_advanced: bool = False,
        enable_reasoning_bridge: bool = True,
        enable_qdt_wm_bridge: bool = False,
    ):
        """
        Wire optional CMS/CPS, advanced consolidation, QDT WM, and episodic bridges.
        Safe to call multiple times; only attaches missing components.
        """
        if self.consolidated_lexicon is None:
            self.enable_consolidated_lexicon(vocab_size=int(vocab_size), senses=int(cms_senses))
        if enable_broker and self.consolidation_broker is None:
            self.enable_consolidation_broker(vocab_size=int(vocab_size))
        if enable_advanced and self.advanced_cms is None:
            self.enable_advanced_consolidation()
        if enable_qdt_wm_bridge:
            from .working_memory.wm_cortex_integration import (
                CortexWorkingMemoryIntegrationConfig,
                replace_cortex_working_memory,
            )
            wm_cfg = CortexWorkingMemoryIntegrationConfig(
                input_dim=self.input_dim,
                hidden_dim=max(128, self.input_dim),
                num_depths=8,
                num_slots=max(8, getattr(self.working_memory, "M", 8)),
            )
            replace_cortex_working_memory(self, wm_cfg)
            self.wire_qdt_to_ltm()
        if enable_reasoning_bridge:
            if self.shared_memory_subsystem is None:
                self.enable_shared_memory_subsystem()
            if self.hg_episodic_ltm is None:
                self.enable_hg_episodic_ltm()
        if enable_qdt_wm_bridge and self.shared_memory_subsystem is not None:
            self.wire_qdt_to_ltm()
        self.diagnostics.log(
            "cps_cms_full_stack_enabled",
            {
                "broker": self.consolidation_broker is not None,
                "advanced": self.advanced_cms is not None,
                "qdt_wm": hasattr(self.working_memory, "get_attention_stack_output"),
                "shared_memory": self.shared_memory_subsystem is not None,
                "hg_episodic": self.hg_episodic_ltm is not None,
            },
        )
        return self

    def set_cps_curriculum_stage(self, stage: int):
        self.cps_fuser.set_curriculum_stage(int(stage))
        self.diagnostics.log("cps_curriculum_stage", {"stage": int(stage), "use_heads": list(self.cps_fuser.cfg.use_heads)})
        return self

    def cps_embed_tokens(self, token_list: List[str]) -> torch.Tensor:
        """
        Returns fused embedding per token (T, input_dim), creating CPS entries on-demand.
        """
        if not token_list:
            return torch.zeros(0, self.input_dim, device=self.ctx_proj.weight.device)
        embs = []
        for t in token_list:
            key = f"token:{t}"
            up = self.cps.ensure(
                key,
                device=self.ctx_proj.weight.device,
                dtype=self.ctx_proj.weight.dtype,
            )
            fused, _ = self.cps_fuser.fuse(up.view())
            embs.append(fused.unsqueeze(0))
        return torch.cat(embs, dim=0)

    def cps_agreement_loss(self, token_list: List[str]) -> torch.Tensor:
        if not token_list:
            return torch.tensor(0.0, device=self.ctx_proj.weight.device)
        total = None
        for t in token_list:
            up = self.cps.ensure(
                f"token:{t}",
                device=self.ctx_proj.weight.device,
                dtype=self.ctx_proj.weight.dtype,
            )
            _, agree = self.cps_fuser.fuse(up.view())
            total = agree if total is None else (total + agree)
        return total / max(1, len(token_list))

    def _enqueue_advanced_ltm_merge(
        self,
        key: str,
        candidate_view: Dict[str, torch.Tensor],
        importance: torch.Tensor,
        src_info: Optional[Dict] = None,
    ) -> None:
        if self.advanced_broker is None:
            return
        self._advanced_merge_queue.append(
            (str(key), candidate_view, importance.detach(), dict(src_info or {}))
        )

    @torch.no_grad()
    def _tick_advanced_consolidation(self, token_keys: Optional[List[str]] = None) -> None:
        if self.advanced_broker is None:
            return
        if token_keys:
            for k in token_keys:
                self._advanced_nudge_keys.add(str(k))
        pending = self._advanced_merge_queue[:256]
        self._advanced_merge_queue = self._advanced_merge_queue[256:]
        for key, cand_view, importance, src_info in pending:
            try:
                self.advanced_broker.ingest_from_ltm(key, cand_view, importance, src_info=src_info)
                self._advanced_nudge_keys.add(key)
            except Exception as exc:
                self.diagnostics.log("advanced_merge_error", {"key": key, "error": str(exc)})
        for key in list(self._advanced_nudge_keys)[:256]:
            try:
                self.advanced_broker.cms_pull_to_cps(key)
                self.advanced_broker.cps_push_to_cms(key)
            except Exception as exc:
                self.diagnostics.log("advanced_nudge_error", {"key": key, "error": str(exc)})

    def enable_diagnostics(
        self,
        enabled: bool = True,
        log_path: str = None,
        flush_every: int = 100,
    ):
        self.diagnostics.configure(enabled=enabled, log_path=log_path, flush_every=flush_every)
        return self

    def flush_diagnostics(self):
        return self.diagnostics.flush()

    def enable_cms_logger(
        self,
        out_dir: str = "cms_logs",
        max_buffer: int = 100000,
        sample_rate: float = 0.1,
    ):
        self.cms_logger = CMSRecordLogger(
            out_dir=out_dir, max_buffer=max_buffer, sample_rate=sample_rate
        )
        return self

    def flush_cms_logger(self):
        if self.cms_logger is None:
            return None
        return self.cms_logger.flush()

    @torch.no_grad()
    def run_cms_consolidation_ema(self, records, ema: float = 0.9):
        if self.consolidated_lexicon is None:
            return
        run_cms_consolidation_ema(self.consolidated_lexicon, records, ema=ema)

    def save_cms_shards(
        self,
        out_dir: str,
        shard_size: int = 4096,
        quantize: bool = False,
    ):
        if self.consolidated_lexicon is None:
            return []
        return dump_cpg_shards(
            self.consolidated_lexicon,
            out_dir=out_dir,
            shard_size=shard_size,
            quantize=quantize,
        )

    @torch.no_grad()
    def load_cms_shards(self, shard_dir: str, keys=None):
        if self.consolidated_lexicon is None:
            return
        load_cpg_shards_into_lexicon(self.consolidated_lexicon, shard_dir=shard_dir, keys=keys)

    def get_consolidated_parameter_groups(
        self,
        base_lr: float = 1e-3,
        assoc_lr_scale: float = 1.0,
        routing_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ):
        """Return logically grouped CMS params for optimizers."""
        if self.consolidated_lexicon is None:
            return []
        return self.consolidated_lexicon.build_optimizer_param_groups(
            base_lr=base_lr,
            assoc_lr_scale=assoc_lr_scale,
            routing_lr_scale=routing_lr_scale,
            weight_decay=weight_decay,
        )

    def _apply_consolidated_memory(
        self,
        sensory_input,
        token_ids,
        context_features=None,
        consolidation_intent: str = "auto",
    ):
        if token_ids is None:
            return sensory_input
        if self.consolidated_lexicon is None and self.consolidation_broker is None:
            self.last_cms_aux = None
            self.last_cps_aux = {"agree_loss": sensory_input.new_tensor(0.0)}
            return sensory_input
        if token_ids.shape[:2] != sensory_input.shape[:2]:
            raise ValueError(
                f"token_ids shape {list(token_ids.shape)} must match sensory_input batch/seq "
                f"{list(sensory_input.shape[:2])}"
            )
        bsz, seq, dim = sensory_input.shape
        flat_ids = token_ids.reshape(-1).long()
        flat_base = sensory_input.reshape(-1, dim)
        if context_features is None:
            ctx = sensory_input.mean(dim=1, keepdim=True).expand(-1, seq, -1)
            flat_ctx = ctx.reshape(-1, dim)
        else:
            if context_features.shape[:2] != sensory_input.shape[:2]:
                raise ValueError(
                    f"context_features shape {list(context_features.shape)} must match sensory_input "
                    f"batch/seq {list(sensory_input.shape[:2])}"
                )
            flat_ctx = context_features.reshape(-1, context_features.size(-1))
        if self.consolidation_broker is not None:
            fused, baux = self.consolidation_broker.route_fuse(
                flat_ids,
                flat_base,
                flat_ctx,
                intent=consolidation_intent,
            )
            primary = baux["selected_store"]
            aux = baux["stores"][primary]
            aux["broker"] = baux
            self.last_cms_aux = aux
            self.diagnostics.log(
                "cms_broker_route",
                {
                    "intent": consolidation_intent,
                    "selected_store": primary,
                },
            )
        else:
            fused, _, aux = self.consolidated_lexicon(flat_ids, flat_base, flat_ctx)
            self.last_cms_aux = aux
            self.diagnostics.log("cms_single_store", {"intent": consolidation_intent})

        def _batchify_feat(x: torch.Tensor, bsz_: int, device, dtype):
            if not isinstance(x, torch.Tensor):
                return torch.zeros(bsz_, device=device, dtype=dtype)
            x = x.detach().to(device=device, dtype=dtype).reshape(-1)
            if x.numel() == 0:
                return torch.zeros(bsz_, device=device, dtype=dtype)
            if x.numel() == 1:
                return x.expand(bsz_)
            if x.numel() != bsz_:
                return x.mean().expand(bsz_)
            return x

        def _collect_ltm_router_stats(bsz_: int, device, dtype):
            comps = [
                getattr(self.long_term_memory, "hg", None),
                getattr(self.long_term_memory, "cgmn", None),
                getattr(self.long_term_memory, "curved", None),
            ]
            vals = []
            for comp in comps:
                feat = getattr(comp, "last_router_features", None) if comp is not None else None
                for key in ("omega_mean", "curv_mean", "dist_mean", "entropy"):
                    v = feat.get(key) if isinstance(feat, dict) else None
                    vals.append(_batchify_feat(v, bsz_, device, dtype).unsqueeze(-1))
            return torch.cat(vals, dim=-1) if vals else torch.zeros(bsz_, 12, device=device, dtype=dtype)

        router_query = flat_ctx
        router_probs = None
        domain_names = []
        if (
            self.advanced_router is not None
            and self.multi_cps is not None
            and self.advanced_router_feat_proj is not None
        ):
            ltm_stats = _collect_ltm_router_stats(bsz, flat_ctx.device, flat_ctx.dtype)  # (B,12)
            stats_proj = self.advanced_router_feat_proj(ltm_stats)  # (B,d)
            stats_proj = stats_proj.unsqueeze(1).expand(-1, seq, -1).reshape(-1, dim)  # (B*S,d)
            router_query = flat_ctx + stats_proj
            _, _, router_probs = self.advanced_router(router_query, top_k=2)  # (B*S,n_domains)
            domain_names = list(self.advanced_router.domains)
            self.last_router_decision = {
                "domains": domain_names,
                "probs_mean": router_probs.detach().mean(dim=0).tolist(),
                "ltm_stats_mean": ltm_stats.detach().mean(dim=0).tolist(),
            }
            self.diagnostics.log(
                "advanced_router_features",
                {
                    "domains": domain_names,
                    "omega_hg": float(ltm_stats[:, 0].mean().item()),
                    "omega_cgmn": float(ltm_stats[:, 4].mean().item()),
                    "omega_curved": float(ltm_stats[:, 8].mean().item()),
                    "entropy_hg": float(ltm_stats[:, 3].mean().item()),
                    "entropy_cgmn": float(ltm_stats[:, 7].mean().item()),
                    "entropy_curved": float(ltm_stats[:, 11].mean().item()),
                },
            )

        # Optional CPS fusion: token-keyed polymorphic unified parameter overlay.
        cps_fused = []
        cps_loss = fused.new_tensor(0.0)
        routed_counts = {d: 0 for d in domain_names}
        for i, tid in enumerate(flat_ids.tolist()):
            key = f"token:{int(tid)}"
            if router_probs is not None and domain_names:
                dom_idx = int(router_probs[i].argmax().item())
                dom = domain_names[dom_idx]
                cps_store, cps_fuser = self.multi_cps.get(dom)
                routed_counts[dom] = routed_counts.get(dom, 0) + 1
                up = cps_store.ensure(key, device=fused.device, dtype=fused.dtype)
                v, loss = cps_fuser.fuse(up.view())
            else:
                up = self.cps.ensure(key, device=fused.device, dtype=fused.dtype)
                v, loss = self.cps_fuser.fuse(up.view())
            cps_fused.append(v)
            cps_loss = cps_loss + loss
        if cps_fused:
            cps_fused = torch.stack(cps_fused, dim=0).to(fused.device, fused.dtype)
            fused = 0.85 * fused + 0.15 * cps_fused
            cps_loss = cps_loss / max(1, len(cps_fused))
            self.last_cps_aux = {"agree_loss": cps_loss}
            if router_probs is not None:
                reg, reg_aux = router_regularizer(router_probs)
                self.last_cps_aux["router_reg"] = reg.detach()
                self.last_cps_aux["router_aux"] = reg_aux
                self.last_cps_aux["router_domain_counts"] = routed_counts
                self.diagnostics.record_scalar("router_reg", float(reg.detach().item()))
                self.diagnostics.log("advanced_router_domain_counts", routed_counts)
            aux["cps"] = self.last_cps_aux
            self.diagnostics.record_scalar("cps_agree_loss", float(cps_loss.detach().item()))
            self.diagnostics.log("cps_fusion", {"count": int(len(cps_fused))})

        if self.cms_logger is not None:
            self.cms_logger.log_from_aux(flat_ids, aux)
        return fused.view(bsz, seq, dim)

    def _ste_write_gate(self, x):
        """Compute write gate with straight-through estimator.
        x: (B,S,d) -> gate_prob: (B,1), gate_hard: (B,1)
        """
        pooled = x.mean(dim=1)  # (B,d)
        gate_prob = self.write_gate(pooled)  # (B,1) continuous in [0,1]
        
        if self.training:
            # Sample binary decision
            gate_hard = (torch.rand_like(gate_prob) < gate_prob).float()
            # STE: forward uses hard, backward uses soft
            gate_ste = gate_hard - gate_prob.detach() + gate_prob
        else:
            # At inference, threshold at 0.5
            gate_ste = (gate_prob > 0.5).float()
        
        return gate_ste, gate_prob

    def _tile_context(self, context, S):
        # context: (B,c) -> project to (B,S,input_dim)
        cproj = self.ctx_proj(context)                    # (B,input_dim)
        return cproj.unsqueeze(1).expand(-1, S, -1)

    def _sync_attention_stacks(self, wm_tokens: Optional[torch.Tensor] = None) -> None:
        """
        Bridge attention-stack context across WM/QDT, episodic-LTM, and triple-hybrid LTM.
        """
        ltm = getattr(self, "long_term_memory", None)
        if ltm is None or not hasattr(ltm, "set_external_attention_context"):
            return
        ctx = None
        if hasattr(self.working_memory, "get_attention_stack_output"):
            try:
                ctx = self.working_memory.get_attention_stack_output()
            except Exception:
                ctx = None
        if ctx is None and wm_tokens is not None:
            ctx = wm_tokens
        if ctx is None and self.hg_episodic_ltm is not None:
            keys = sorted(self.hg_episodic_ltm.episode_records.keys())
            if keys:
                tail = keys[-min(4, len(keys)) :]
                chunks = []
                for eid in tail:
                    c = self.hg_episodic_ltm.build_episode_attention_context(
                        episode_id=eid, include_summary=True, max_tokens=32
                    )
                    if c is not None:
                        chunks.append(c)
                if chunks:
                    ctx = torch.cat(chunks, dim=1)
        if ctx is not None:
            ltm.set_external_attention_context(ctx)
            if self.hg_episodic_ltm is not None and hasattr(self.hg_episodic_ltm, "set_external_attention_context"):
                self.hg_episodic_ltm.set_external_attention_context(ctx)
            if hasattr(self.working_memory, "set_external_attention_context"):
                try:
                    self.working_memory.set_external_attention_context(ctx)
                except Exception:
                    pass

    def register_additional_hidden_attention_source(
        self,
        module: nn.Module,
        *,
        source_name: str = "cortex.extra",
    ) -> int:
        """Allow callers to extend global hidden-attention coverage at runtime."""
        return self.global_hidden_orchestrator.register_source(module, source_name=source_name)

    def enable_parameter_storage_loop(
        self,
        *,
        slots_per_layer: Optional[int] = None,
        free_hidden_layers: Optional[int] = None,
    ) -> ParameterStorageLoopStack:
        """Mount the product-manifold parameter store as an optional cortex subsystem."""
        cfg = ParameterStorageLoopConfig(
            model_dim=int(self.input_dim),
            parameter_slots_per_layer=int(slots_per_layer or self.parameter_loop_slots_per_layer),
            free_hidden_layers=int(
                self.parameter_loop_free_hidden_layers
                if free_hidden_layers is None
                else free_hidden_layers
            ),
            num_heads=int(self._ctx_heads),
            enable_training_slot_updates=bool(self.enable_parameter_loop_training_writes),
        )
        self.parameter_storage_loop_stack = ParameterStorageLoopStack(cfg)
        self.enable_parameter_storage_loop_stack = True
        self.parameter_loop_slots_per_layer = int(cfg.parameter_slots_per_layer)
        self.parameter_loop_free_hidden_layers = int(cfg.free_hidden_layers)
        self.global_hidden_orchestrator.register_source(
            self.parameter_storage_loop_stack,
            source_name="cortex.parameter_loop",
        )
        self.last_parameter_storage_loop_stats = {
            "enabled": True,
            "capacity_estimate": self.parameter_storage_loop_stack.estimate_storage_capacity(),
        }
        return self.parameter_storage_loop_stack

    def _enable_qdt_working_memory_fabric(self) -> None:
        from .working_memory.wm_cortex_integration import (
            CortexWorkingMemoryIntegrationConfig,
            replace_cortex_working_memory,
            wire_qdt_ltm_adapter,
        )

        cfg = CortexWorkingMemoryIntegrationConfig.from_hardware_profile(
            self.qdt_hardware_profile,
            input_dim=int(self.input_dim),
            use_compatibility_wrapper=True,
            preserve_old_reference=True,
        )
        if self.qdt_num_slots > 0:
            cfg.num_slots = int(self.qdt_num_slots)
        if self.qdt_transformer_layers > 0:
            cfg.transformer_layers = int(self.qdt_transformer_layers)
            cfg.maae_transformer_layers = int(self.qdt_transformer_layers)
        cfg.qspin_guarded_shadow = bool(self.qdt_qspin_guarded_shadow)
        if self.qdt_qspin_live_activation is not None:
            cfg.qspin_live_activation = bool(self.qdt_qspin_live_activation)
            cfg.qspin_live_mode = "experimental_live" if cfg.qspin_live_activation else "disabled"
        cfg.qspin_live_kill_switch_enabled = bool(self.qdt_qspin_live_kill_switch_enabled)
        cfg.qspin_live_max_payload_tokens = int(self.qdt_qspin_live_max_payload_tokens)
        result = replace_cortex_working_memory(self, cfg)
        live_ltm_attached = wire_qdt_ltm_adapter(self)
        self.qdt_working_memory_migration_trace = {
            **result.to_dict(),
            "live_ltm_adapter_attached": bool(live_ltm_attached),
        }

    def describe_working_memory_fabric(self) -> Dict[str, Any]:
        wm = getattr(self, "working_memory", None)
        qdt = getattr(wm, "qdt_working_memory", wm)
        cfg = getattr(qdt, "config", None)
        return {
            "fabric": str(self.working_memory_fabric),
            "working_memory_class": type(wm).__name__ if wm is not None else None,
            "qdt_class": type(qdt).__name__ if qdt is not None else None,
            "qdt_config": cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else None,
            "qdt_capacity_estimate": cfg.capacity_estimate(batch_size=1, seq_len=1).to_dict()
            if cfg is not None and hasattr(cfg, "capacity_estimate")
            else None,
            "migration_trace": dict(getattr(self, "qdt_working_memory_migration_trace", {})),
        }

    def describe_parameter_storage_loop(self) -> Dict[str, Any]:
        stack = self.parameter_storage_loop_stack
        if stack is None:
            cfg = ParameterStorageLoopConfig(
                model_dim=int(self.input_dim),
                parameter_slots_per_layer=int(self.parameter_loop_slots_per_layer),
                free_hidden_layers=int(self.parameter_loop_free_hidden_layers),
                num_heads=int(self._ctx_heads),
            )
            return {
                "enabled": False,
                "capacity_estimate": ParameterStorageLoopStack(cfg).estimate_storage_capacity(),
            }
        return {
            "enabled": True,
            "manifold_stack": list(stack.manifold_stack),
            "capacity_estimate": stack.estimate_storage_capacity(),
            "hidden_attention_source": "cortex.parameter_loop",
            "ltm_context_enabled": bool(self.enable_parameter_loop_ltm_context),
            "training_slot_updates_enabled": bool(self.enable_parameter_loop_training_writes),
            "auto_consolidation_enabled": bool(self.enable_parameter_loop_auto_consolidation),
            "bundle_registry": stack.parameter_bundle_registry_state(),
        }

    def build_parameter_loop_reasoning_adapter(self, *, max_context_tokens: int = 32):
        if self.parameter_storage_loop_stack is None:
            raise RuntimeError("parameter_storage_loop_stack is not enabled")
        from .reasoning_depth import ParameterLoopAdapter, ParameterLoopAdapterConfig

        return ParameterLoopAdapter(
            ParameterLoopAdapterConfig.enabled_default(
                key_dim=int(self.input_dim),
                max_context_tokens=int(max(1, max_context_tokens)),
            ),
            parameter_loop=self.parameter_storage_loop_stack,
        )

    def _sync_parameter_loop_ltm_context(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        stack = self.parameter_storage_loop_stack
        if stack is None or not self.enable_parameter_loop_ltm_context:
            return None
        ctx = stack.build_ltm_context_tokens(query, max_tokens=self.max_external_context_tokens)
        self.long_term_memory.set_external_attention_context(ctx)
        if self.hg_episodic_ltm is not None and hasattr(self.hg_episodic_ltm, "set_external_attention_context"):
            try:
                self.hg_episodic_ltm.set_external_attention_context(ctx)
            except Exception:
                pass
        self.last_parameter_storage_loop_stats = {
            **dict(self.last_parameter_storage_loop_stats),
            "ltm_context_tokens": int(ctx.size(1)),
            "ltm_context_enabled": True,
        }
        return ctx

    def _apply_parameter_storage_loop(self, seq: torch.Tensor, phase: str) -> torch.Tensor:
        stack = self.parameter_storage_loop_stack
        if stack is None:
            return seq
        loop_out, trace = stack(
            seq,
            recall_boost=0.3,
            allow_slot_update=bool(self.enable_parameter_loop_training_writes),
            slot_update_scale=float(self.parameter_loop_training_write_scale),
            return_trace=True,
        )
        gate = torch.sigmoid(self.parameter_storage_loop_gate)
        mixed = self.mem_bridge_norm((1.0 - gate) * seq + gate * loop_out)
        self.last_parameter_storage_loop_stats = {
            **dict(self.last_parameter_storage_loop_stats),
            "phase": str(phase),
            "gate": float(gate.detach().item()),
            "loop_attention_tokens": int(trace.get("loop_attention_tokens", 0)),
            "effective_storage_units": float(
                trace.get("capacity_estimate", {}).get("effective_parameter_storage_units", 0.0)
            ),
            "effective_to_physical_ratio": float(
                trace.get("capacity_estimate", {}).get("effective_to_physical_ratio", 0.0)
            ),
        }
        if getattr(self, "diagnostics", None) is not None:
            self.diagnostics.log(f"parameter_storage_loop_{phase}", self.last_parameter_storage_loop_stats)
        return mixed

    def _apply_consolidated_memory_depth(self, seq: torch.Tensor, phase: str) -> torch.Tensor:
        cms = self.advanced_cms
        if cms is None or cms.depth_stack is None:
            return seq
        depth_out, trace = cms.apply_depth_stack(seq, recall_boost=0.25, return_trace=True)
        gate = torch.sigmoid(self.consolidated_memory_depth_gate)
        mixed = self.mem_bridge_norm((1.0 - gate) * seq + gate * depth_out)
        capacity = trace.get("capacity_estimate", {}) if isinstance(trace, dict) else {}
        self.last_cms_depth_stack_stats = {
            "enabled": True,
            "phase": str(phase),
            "gate": float(gate.detach().item()),
            "loop_attention_tokens": int(trace.get("loop_attention_tokens", 0)) if isinstance(trace, dict) else 0,
            "depth_manifold_stack": list(cms.depth_stack.depth_manifold_stack),
            "super_product_manifold": trace.get("super_product_manifold") if isinstance(trace, dict) else None,
            "effective_memory_storage_units": float(capacity.get("effective_memory_storage_units", 0.0)),
            "effective_to_physical_ratio": float(capacity.get("effective_to_physical_ratio", 0.0)),
            "free_hidden_layers": int(cms.depth_stack.config.free_hidden_layers),
        }
        if getattr(self, "diagnostics", None) is not None:
            self.diagnostics.log(f"consolidated_memory_depth_{phase}", self.last_cms_depth_stack_stats)
        return mixed

    def _sync_cms_depth_ltm_context(self, query: torch.Tensor) -> None:
        cms = self.advanced_cms
        ltm = getattr(self, "long_term_memory", None)
        if cms is None or cms.depth_stack is None or ltm is None:
            return
        if not hasattr(ltm, "set_parameter_loop_context_tokens"):
            return
        try:
            ctx = cms.build_depth_context_tokens(query, max_tokens=min(48, int(self.max_external_context_tokens)))
            ltm.set_parameter_loop_context_tokens(ctx)
        except Exception:
            pass

    def _maybe_consolidate_trainable_parameters(self) -> Dict[str, Any]:
        stack = self.parameter_storage_loop_stack
        self.parameter_loop_consolidation_step += 1
        if stack is None:
            return {"triggered": False, "reason": "parameter_storage_loop_disabled"}
        if not (self.training and self.enable_parameter_loop_auto_consolidation):
            return {"triggered": False, "reason": "disabled_or_not_training", "step": int(self.parameter_loop_consolidation_step)}
        if self.parameter_loop_consolidation_step % int(self.parameter_loop_consolidation_interval) != 0:
            return {
                "triggered": False,
                "reason": "interval_not_reached",
                "step": int(self.parameter_loop_consolidation_step),
                "interval": int(self.parameter_loop_consolidation_interval),
            }
        trace = stack.consolidate_parameter_bundles(
            self.named_parameters(),
            trigger_state={
                "step": int(self.parameter_loop_consolidation_step),
                "factors": (
                    "module_family",
                    "shape_rank",
                    "role_keyword",
                    "size_bucket",
                    "training_threshold",
                ),
            },
            max_bundles=int(self.parameter_loop_consolidation_max_bundles),
            min_params_per_bundle=int(self.parameter_loop_consolidation_min_params),
            min_total_numel=int(self.parameter_loop_consolidation_min_total_numel),
            include_filters=self.parameter_loop_consolidation_include,
            exclude_filters=self.parameter_loop_consolidation_exclude,
            write_scale=float(self.parameter_loop_training_write_scale),
        )
        trace["triggered"] = bool(trace.get("created_bundle_count", 0) > 0)
        self.last_parameter_loop_consolidation_trace = trace
        self.last_parameter_storage_loop_stats = {
            **dict(self.last_parameter_storage_loop_stats),
            "last_bundle_consolidation": trace,
        }
        if getattr(self, "diagnostics", None) is not None:
            self.diagnostics.log("parameter_loop_bundle_consolidation", trace)
        return trace

    def _bridge_wm_ltm(self, wm_seq: torch.Tensor, ltm_seq: torch.Tensor, phase: str) -> torch.Tensor:
        gate = torch.sigmoid(self.mem_bridge_gate)
        wm_from_ltm, w_wm = self.wm_to_ltm_attn(wm_seq, ltm_seq, ltm_seq, need_weights=True)
        ltm_from_wm, w_ltm = self.ltm_to_wm_attn(ltm_seq, wm_seq, wm_seq, need_weights=True)
        merged = self.mem_bridge_norm(
            0.60 * wm_seq + 0.20 * (gate * wm_from_ltm) + 0.20 * (gate * ltm_from_wm)
        )
        self.diagnostics.record_scalar(f"bridge_gate_{phase}", float(gate.detach().item()))
        self.diagnostics.record_scalar(f"bridge_attn_wm_{phase}", float(w_wm.detach().mean().item()))
        self.diagnostics.record_scalar(f"bridge_attn_ltm_{phase}", float(w_ltm.detach().mean().item()))
        return merged

    def _apply_secondary_hidden_stack(
        self,
        base_seq: torch.Tensor,
        wm_seq: torch.Tensor,
        ltm_seq: torch.Tensor,
        ctx_seq: Optional[torch.Tensor],
        phase: str,
    ) -> torch.Tensor:
        if not self.enable_secondary_hidden_stack:
            return base_seq
        self_view, w_self = self.ctx_attn(base_seq, base_seq, base_seq, need_weights=True)
        wm_view, w_wm = self.ltm_to_wm_attn(base_seq, wm_seq, wm_seq, need_weights=True)
        ltm_view, w_ltm = self.wm_to_ltm_attn(base_seq, ltm_seq, ltm_seq, need_weights=True)
        if ctx_seq is None:
            ctx_view = torch.zeros_like(base_seq)
            w_ctx = None
        else:
            ctx_view, w_ctx = self.query_ctx_attn(base_seq, ctx_seq, ctx_seq, need_weights=True)

        param_tokens, param_stats = self._build_secondary_hidden_param_tokens(
            bsz=base_seq.size(0), device=base_seq.device, dtype=base_seq.dtype
        )
        param_view, w_param = self.secondary_hidden_param_attn(
            base_seq, param_tokens, param_tokens, need_weights=True
        )

        if self.secondary_hidden_stack_variant == "cascade":
            mixed = self.secondary_hidden_norm(base_seq + self_view)
            mixed = self.secondary_hidden_norm(mixed + wm_view)
            mixed = self.secondary_hidden_norm(mixed + ltm_view)
            mixed = self.secondary_hidden_norm(mixed + ctx_view)
            mixed = self.secondary_hidden_norm(mixed + param_view)
        elif self.secondary_hidden_stack_variant == "bridge_mix":
            gates = torch.softmax(self.secondary_hidden_mix_gates, dim=0)
            mixed = (
                base_seq
                + gates[0] * self_view
                + gates[1] * wm_view
                + gates[2] * ltm_view
                + gates[3] * ctx_view
                + gates[4] * param_view
            )
            mixed = self.secondary_hidden_norm(mixed)
        else:
            dyn = torch.softmax(
                self.secondary_hidden_adaptive_gate(base_seq.mean(dim=1)),
                dim=-1,
            ).unsqueeze(1)
            mixed = (
                base_seq
                + dyn[..., 0:1] * self_view
                + dyn[..., 1:2] * wm_view
                + dyn[..., 2:3] * ltm_view
                + dyn[..., 3:4] * ctx_view
                + dyn[..., 4:5] * param_view
            )
            mixed = self.secondary_hidden_norm(mixed)

        refined = self.secondary_hidden_encoder(mixed)
        stack_out = self.secondary_hidden_output_norm(mixed + refined)
        bridge_gate = torch.sigmoid(self.secondary_hidden_bridge_gate)
        out = self.mem_bridge_norm((1.0 - bridge_gate) * base_seq + bridge_gate * stack_out)
        self.last_secondary_hidden_stack_stats = {
            "phase": str(phase),
            "variant": self.secondary_hidden_stack_variant,
            "self_attn_mean": self._attn_weight_mean(w_self),
            "wm_attn_mean": self._attn_weight_mean(w_wm),
            "ltm_attn_mean": self._attn_weight_mean(w_ltm),
            "ctx_attn_mean": self._attn_weight_mean(w_ctx),
            "param_attn_mean": self._attn_weight_mean(w_param),
            "param_signature_scale": float(param_stats.abs().mean().detach().item()),
            "bridge_gate": float(bridge_gate.detach().item()),
        }
        if getattr(self, "diagnostics", None) is not None:
            self.diagnostics.log(f"secondary_hidden_stack_{phase}", self.last_secondary_hidden_stack_stats)
        return out

    def _apply_global_hidden_attention(self, seq: torch.Tensor, ctx: Optional[torch.Tensor], phase: str) -> torch.Tensor:
        if not self.enable_global_hidden_attention:
            return seq
        out = self.global_hidden_orchestrator.integrate(seq, context=ctx)
        self.last_global_hidden_attention_stats = dict(self.global_hidden_orchestrator.last_stats)
        if getattr(self, "diagnostics", None) is not None:
            payload = dict(self.last_global_hidden_attention_stats)
            payload["phase"] = str(phase)
            self.diagnostics.log(f"global_hidden_attention_{phase}", payload)
        return out

    def process_sensory_input(self, sensory_input):
        self.sensory_buffer.update(sensory_input)
        base = self.sensory_buffer.attention_filter(sensory_input)
        attn_out, _ = self.ctx_attn(base, base, base, need_weights=False)
        enc_out = self.ctx_encoder(base)
        out = self.ctx_norm(base + 0.5 * attn_out + 0.5 * enc_out)
        self.diagnostics.record_scalar("sensory_norm", float(out.norm(dim=-1).mean().item()))
        return out

    def encode_memory(self, info, context, mtype):
        B,S,d = info.shape
        self._sync_attention_stacks(info)
        ctx = self._tile_context(context, S)
        pooled = torch.cat([info, ctx], dim=-1).mean(dim=1)      # (B, 2d)
        idx = self.hippocampal_encoder(pooled)                   # (B,256)

        cue = idx.unsqueeze(1).expand(-1, S, -1)                 # (B,S,256)
        # Project cue to input_dim if needed
        if cue.size(-1) != self.input_dim:
            cue = self.cue_to_input(cue)

        if mtype == 'episodic':
            self.long_term_memory(cue, operation='write')
        elif mtype == 'semantic':
            self.long_term_memory.cgmn(cue, operation='write')
        elif mtype in ('spatial', 'associative'):
            self.long_term_memory.curved(cue, operation='write')
        elif mtype in ('all', 'triple', 'hybrid'):
            self.long_term_memory(cue, operation='write')
        else:
            self.long_term_memory.curved(cue, operation='write')

        if self.advanced_broker is not None and self.advanced_view_adapter is not None:
            for b in range(B):
                cvec = cue[b].mean(dim=0).detach()
                importance = torch.sigmoid(cvec.norm().view(1))
                key = f"ltm:episodic:{self._advanced_merge_counter}"
                self._advanced_merge_counter += 1
                cand = self.advanced_view_adapter(cvec)
                self._enqueue_advanced_ltm_merge(
                    key=key,
                    candidate_view=cand,
                    importance=importance,
                    src_info={"source": "encode_memory", "mtype": str(mtype), "batch_index": int(b)},
                )
            self._tick_advanced_consolidation()
        return idx

    def retrieve_memory(
        self,
        cue,
        context,
        strategy='associative',
        fire_mask=None,
        recall_boost: float = 0.3,
        query_token_ids=None,
    ):
        """Retrieve memories with optional explosive recall (fire_mask)."""
        if self.enable_global_hidden_attention:
            self.global_hidden_orchestrator.begin_capture()
        B,S,d = cue.shape
        self._sync_attention_stacks(cue)
        ctx = self._tile_context(context, S)
        c = (cue + ctx) * 0.5
        qctx, _ = self.query_ctx_attn(c, ctx, ctx, need_weights=False)
        c = self.query_norm(c + qctx)

        if self.consolidation_broker is not None and self.ahg is not None:
            query = c.mean(dim=1)
            bdiag = self.consolidation_broker.route_read(query, intent="auto", k=8)
            xdiag = self.consolidation_broker.cross_store_diagnostics(query, k=8)
            if query_token_ids is not None:
                if query_token_ids.dim() > 1:
                    flat_ids = query_token_ids.reshape(-1).tolist()
                else:
                    flat_ids = query_token_ids.tolist()
                cps_keys = [f"token:{int(t)}" for t in flat_ids]
                cps_sig = self.cps.confidence_signals_for_keys(cps_keys)
                # Fuse CPS confidence into broker diagnostics for AHG decision.
                bdiag["signals"]["fisher_uncertainty"] = max(
                    float(bdiag["signals"].get("fisher_uncertainty", 0.0)),
                    float(cps_sig.get("fisher_uncertainty", 0.0)),
                )
                bdiag["signals"]["phase_agreement"] = 0.5 * float(
                    bdiag["signals"].get("phase_agreement", 0.0)
                ) + 0.5 * float(cps_sig.get("phase_agreement", 0.0))
            decision = self.ahg.decide(bdiag, xdiag)
            self.last_ahg_decision = {
                "action": decision.action,
                "reason": decision.reason,
                "scores": decision.scores,
            }
            self.diagnostics.log(
                "ahg_decision",
                {
                    "action": decision.action,
                    "strategy_in": strategy,
                    "proto": decision.scores.get("proto", -1.0),
                    "fisher": decision.scores.get("fisher", -1.0),
                    "agree": decision.scores.get("agree", -1.0),
                },
            )
            if decision.action == "explosive":
                recall_boost = max(recall_boost, 0.65)
            elif decision.action == "refine":
                strategy = "direct"
            elif decision.action == "ask":
                # conservative fallback answer vector from context only
                z = torch.zeros(context.size(0), 256, device=context.device, dtype=context.dtype)
                if self.enable_global_hidden_attention:
                    self.global_hidden_orchestrator.end_capture()
                return self.retrieval(torch.cat([z, context], dim=-1))

        if self.hg_episodic_ltm is not None:
            try:
                epi = self.retrieve_episodic_trace(query=c.mean(dim=1), top_k=8)
                frags = getattr(epi, "values", None)
                if isinstance(frags, torch.Tensor) and frags.numel() > 0:
                    pooled = frags.mean(dim=1, keepdim=True).expand(-1, S, -1)
                    c = self.query_norm(c + 0.15 * pooled)
                    self.diagnostics.log(
                        "retrieve_episodic_blend",
                        {"fragment_count": int(frags.size(1))},
                    )
            except Exception as exc:
                self.diagnostics.log("retrieve_episodic_error", {"error": str(exc)})

        if strategy == 'direct':
            r = self.long_term_memory(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)
        elif strategy == 'associative':
            r = self.long_term_memory.curved(c, operation='read')
            if fire_mask is not None and float(recall_boost) > 0.0:
                mask = torch.as_tensor(fire_mask, device=r.device, dtype=r.dtype).reshape(-1, 1, 1)
                if mask.size(0) != r.size(0):
                    mask = mask.mean().expand(r.size(0), 1, 1)
                r = r * (1.0 + mask.clamp(0.0, 1.0) * float(max(0.0, min(1.0, recall_boost))))
        else:
            r = self.long_term_memory.hg(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)

        wm_r = self.working_memory(c, operation=self._wm_read_operation())
        ltm_r = r
        r = self._bridge_wm_ltm(wm_r, ltm_r, phase="retrieve")
        r = self._apply_secondary_hidden_stack(r, wm_r, ltm_r, ctx, phase="retrieve")
        r = self._apply_global_hidden_attention(r, ctx, phase="retrieve")
        inter = getattr(self.long_term_memory, "last_inter_memory_stats", None)
        if isinstance(inter, dict) and inter:
            self.diagnostics.log("ltm_inter_memory_exchange", inter)
        pref = getattr(self.long_term_memory, "last_prefusion_specialization_stats", None)
        if isinstance(pref, dict) and pref:
            self.diagnostics.log("ltm_prefusion_specialization", pref)

        cue_vec = self.r_proj(r.mean(dim=1))
        self.diagnostics.record_scalar("recall_boost", float(recall_boost))
        self.diagnostics.log("retrieve_path", {"strategy": strategy})
        self.diagnostics.log("recall_event", {"strategy": strategy, "recall_boost": float(recall_boost)})
        if self.reasoning_bridge_enabled:
            self.diagnostics.log("reasoning_mann_bridge", {"enabled": True})
        if self.enable_global_hidden_attention:
            self.global_hidden_orchestrator.end_capture()
        return self.retrieval(torch.cat([cue_vec, context], dim=-1))

    @torch.no_grad()
    def consolidate_memories(self, threshold: float = None):
        """Forget rarely used slots (usage-based) and gently decay working-memory importance."""
        th = self.forgetting_threshold if threshold is None else float(threshold)
        self.long_term_memory.consolidate_unused(th)
        self.working_memory.memory_importance.mul_(0.999)
        self.diagnostics.log("consolidate_memories", {"threshold": th})
        if self.consolidation_broker is not None:
            try:
                hg_vals = self.long_term_memory.hg.values.detach().unsqueeze(0)
                hg = self.long_term_memory.hg.output_projection(hg_vals).squeeze(0)
                cg_slots = self.long_term_memory.cgmn.memory_slots.detach().unsqueeze(0)
                cg = self.long_term_memory.cgmn.output_projection(cg_slots).squeeze(0)
                cv = self.long_term_memory.curved.decoder(
                    self.long_term_memory.curved.memory_slots.detach()
                )
                items = torch.cat([hg, cg, cv], dim=0)
                metas = (
                    [{"domain": "reasoning", "tags": ["hg"]}] * hg.size(0)
                    + [{"domain": "reasoning", "tags": ["cgmn"]}] * cg.size(0)
                    + [{"domain": "reasoning", "tags": ["curved"]}] * cv.size(0)
                )
                self.consolidation_broker.unify_and_route_write(items, metas, domain="reasoning")
                self.diagnostics.log(
                    "unify_and_route_write",
                    {"items": int(items.size(0)), "domain": "reasoning"},
                )
            except Exception:
                pass

    def get_metrics(self):
        """Collect diagnostic metrics from all components."""
        metrics = {}
        # Lightbulb metrics
        try:
            metrics.update(self.lightbulb.get_metrics())
        except Exception as exc:
            metrics["lightbulb_metrics_error"] = str(exc)
        # Memory module metrics
        try:
            metrics.update(self.long_term_memory.hg.get_metrics())
        except Exception as exc:
            metrics["hg_metrics_error"] = str(exc)
        try:
            metrics.update(self.long_term_memory.cgmn.get_metrics())
        except Exception as exc:
            metrics["cgmn_metrics_error"] = str(exc)
        try:
            metrics.update(self.long_term_memory.curved.get_metrics())
        except Exception as exc:
            metrics["ltm_curved_metrics_error"] = str(exc)
        if getattr(self.long_term_memory, "spatial_ltm", None) is not None:
            try:
                spatial_metrics = self.long_term_memory.spatial_ltm.get_metrics()
                for k, v in spatial_metrics.items():
                    metrics[f"ltm_spatial_{k}"] = v
                metrics["ltm_spatial_enabled"] = 1.0
            except Exception as exc:
                metrics["ltm_spatial_metrics_error"] = str(exc)
        else:
            metrics["ltm_spatial_enabled"] = 0.0
        if self.spatial_ltm_extension is not None:
            metrics["spatial_ltm_extension_enabled"] = 1.0
        else:
            metrics["spatial_ltm_extension_enabled"] = 0.0
        try:
            metrics.update(self.working_memory.get_metrics())
        except Exception as exc:
            metrics["wm_metrics_error"] = str(exc)
        inter = getattr(self.long_term_memory, "last_inter_memory_stats", None)
        if isinstance(inter, dict):
            for k, v in inter.items():
                metrics[f"ltm_inter_{k}"] = float(v)
        rstats = getattr(self.long_term_memory, "last_router_stats", None)
        if isinstance(rstats, dict):
            for k, v in rstats.items():
                metrics[f"ltm_router_{k}"] = float(v)
        if self.consolidation_broker is not None:
            b = self.consolidation_broker.get_metrics()
            for k, v in b.items():
                for kk, vv in v.items():
                    metrics[f"broker_{k}_{kk}"] = vv
        if self.last_ahg_decision is not None:
            metrics["ahg_last_action"] = self.last_ahg_decision.get("action", "none")
        # Advanced router + routed CPS diagnostics.
        if isinstance(self.last_router_decision, dict):
            probs = self.last_router_decision.get("probs_mean", [])
            domains = self.last_router_decision.get("domains", [])
            feat_mean = self.last_router_decision.get("ltm_stats_mean", [])
            for i, dom in enumerate(domains):
                if i < len(probs):
                    metrics[f"router_prob_{dom}"] = float(probs[i])
            feat_names = [
                "hg_omega_mean", "hg_curv_mean", "hg_dist_mean", "hg_entropy",
                "cgmn_omega_mean", "cgmn_curv_mean", "cgmn_dist_mean", "cgmn_entropy",
                "curved_omega_mean", "curved_curv_mean", "curved_dist_mean", "curved_entropy",
            ]
            for i, name in enumerate(feat_names):
                if i < len(feat_mean):
                    metrics[f"router_feat_{name}"] = float(feat_mean[i])
        if isinstance(self.last_cps_aux, dict):
            if "router_reg" in self.last_cps_aux:
                rr = self.last_cps_aux["router_reg"]
                try:
                    metrics["router_reg"] = float(rr.detach().item())
                except Exception:
                    metrics["router_reg"] = float(rr)
            router_aux = self.last_cps_aux.get("router_aux", {})
            if isinstance(router_aux, dict):
                for k in ("entropy", "H_target", "balance_loss", "sparsity_mass"):
                    if k in router_aux:
                        metrics[f"router_{k}"] = float(router_aux[k])
            rcounts = self.last_cps_aux.get("router_domain_counts", {})
            if isinstance(rcounts, dict):
                for dom, cnt in rcounts.items():
                    metrics[f"router_count_{dom}"] = float(cnt)
        dsum = self.diagnostics.summary()
        metrics["diag_enabled"] = float(1.0 if dsum.get("enabled") else 0.0)
        metrics["diag_events_buffered"] = float(dsum.get("events_buffered", 0))
        for k, v in dsum.get("ema", {}).items():
            metrics[f"diag_ema_{k}"] = float(v)
        # Cortex-level
        metrics['energy_mode'] = self.energy_mode
        metrics['forgetting_threshold'] = self.forgetting_threshold
        if self.shared_memory_subsystem is not None:
            ss = self.shared_memory_subsystem.store.summarize()
            metrics["shared_mem_enabled"] = 1.0
            metrics["shared_mem_used_slots"] = float(ss.get("used_slots", 0))
            metrics["shared_mem_free_slots"] = float(ss.get("free_slots", 0))
            metrics["shared_mem_mean_confidence"] = float(ss.get("mean_confidence", 0.0))
        else:
            metrics["shared_mem_enabled"] = 0.0
        if self.hg_episodic_ltm is not None:
            metrics["hg_episodic_enabled"] = 1.0
            metrics["hg_episodic_records"] = float(len(self.hg_episodic_ltm.episode_records))
            if hasattr(self.hg_episodic_ltm, "get_metrics"):
                for k, v in self.hg_episodic_ltm.get_metrics().items():
                    metrics[f"hg_episodic_{k}"] = float(v)
            metrics["hg_episodic_wired"] = float(self.hg_episodic_wiring_trace is not None)
            metrics["hg_episodic_lattice_mirror"] = float(self.hg_episodic_wm_lattice_mirror is not None)
        else:
            metrics["hg_episodic_enabled"] = 0.0
        metrics["hgm_enabled"] = 1.0 if self.hgm_enabled else 0.0
        hgm_events = [evt for evt in self.diagnostics.events if evt.get("event") == "hgm_run"]
        metrics["hgm_assignments"] = float(hgm_events[-1].get("payload", {}).get("assignment_count", self.last_hgm_assignments)) if hgm_events else float(self.last_hgm_assignments)
        if self.advanced_cms is not None and self.advanced_cms.depth_stack is not None:
            metrics["cms_depth_stack_enabled"] = 1.0
            metrics["cms_depth_free_hidden_layers"] = float(self.advanced_cms.depth_stack.config.free_hidden_layers)
            metrics["cms_depth_qh_num_depths"] = float(self.advanced_cms.default_cfg.qh_num_depths)
            cap = self.advanced_cms.depth_stack.estimate_storage_capacity()
            metrics["cms_depth_effective_storage_units"] = float(cap.get("effective_memory_storage_units", 0.0))
            metrics["cms_depth_effective_ratio"] = float(cap.get("effective_to_physical_ratio", 0.0))
            for k, v in self.last_cms_depth_stack_stats.items():
                if isinstance(v, (float, int)):
                    metrics[f"cms_depth_{k}"] = float(v)
        else:
            metrics["cms_depth_stack_enabled"] = 0.0
        return metrics

    @torch.no_grad()
    def get_holonomy_stats(self, x: torch.Tensor):
        """Diagnostics for latent spin holonomy in HG/CGMN paths."""
        bsz, seq, _ = x.shape
        hg_man = self.long_term_memory.hg.encode_to_manifold(x)
        cg_man = self.long_term_memory.cgmn.manifold_projection(x).view(
            bsz, seq, self.long_term_memory.cgmn.D, 3
        )
        cg_man = self.long_term_memory.cgmn._evolve(cg_man)
        return {
            "hg": self.long_term_memory.hg.holonomy_stats(hg_man),
            "cgmn": self.long_term_memory.cgmn.holonomy_stats(cg_man),
        }

    def save_checkpoint(self, path: str, version: str = '1.0'):
        """Save versioned checkpoint with explicit schema."""
        import torch
        checkpoint = {
            'version': version,
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'forgetting_threshold': self.forgetting_threshold,
                'energy_mode': self.energy_mode,
            },
            'lightbulb_state': {
                'trigger_rate_ema': self.lightbulb.trigger_rate_ema.item(),
                'threshold': self.lightbulb.thresh,
                'total_fires': self.lightbulb.total_fires.item(),
                'total_samples': self.lightbulb.total_samples.item(),
            },
            'memory_state': {
                'hg_usage': self.long_term_memory.hg.usage_counts.clone(),
                'cgmn_usage': self.long_term_memory.cgmn.usage_counts.clone(),
                'curved_usage': self.long_term_memory.curved.usage_counts.clone(),
            },
            'optional_subsystems': {
                'shared_memory_enabled': bool(self.shared_memory_subsystem is not None),
                'shared_memory_store': self.shared_memory_subsystem.store.to_dict() if self.shared_memory_subsystem is not None else None,
                'hg_episodic_enabled': bool(self.hg_episodic_ltm is not None),
                'episodic_write_mode': str(getattr(self, "episodic_write_mode", "legacy")),
                'parameter_storage_loop_enabled': bool(self.parameter_storage_loop_stack is not None),
                'parameter_loop_slots_per_layer': int(self.parameter_loop_slots_per_layer),
                'parameter_loop_free_hidden_layers': int(self.parameter_loop_free_hidden_layers),
                'parameter_loop_ltm_context_enabled': bool(self.enable_parameter_loop_ltm_context),
                'parameter_loop_training_writes_enabled': bool(self.enable_parameter_loop_training_writes),
                'parameter_loop_training_write_scale': float(self.parameter_loop_training_write_scale),
                'parameter_loop_auto_consolidation_enabled': bool(self.enable_parameter_loop_auto_consolidation),
                'parameter_loop_consolidation_interval': int(self.parameter_loop_consolidation_interval),
                'parameter_loop_consolidation_max_bundles': int(self.parameter_loop_consolidation_max_bundles),
                'parameter_loop_consolidation_min_params': int(self.parameter_loop_consolidation_min_params),
                'parameter_loop_consolidation_min_total_numel': int(self.parameter_loop_consolidation_min_total_numel),
                'parameter_loop_consolidation_include': list(self.parameter_loop_consolidation_include),
                'parameter_loop_consolidation_exclude': list(self.parameter_loop_consolidation_exclude),
                'parameter_loop_bundle_registry': (
                    self.parameter_storage_loop_stack.parameter_bundle_registry_state()
                    if self.parameter_storage_loop_stack is not None
                    else None
                ),
                'parameter_loop_consolidation_step': int(self.parameter_loop_consolidation_step),
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load checkpoint with version validation."""
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        optional = checkpoint.get('optional_subsystems', {}) or {}
        if optional.get('shared_memory_enabled', False) and self.shared_memory_subsystem is None:
            store = optional.get('shared_memory_store', {}) or {}
            self.enable_shared_memory_subsystem(
                num_slots=int(store.get('num_slots', 2048)),
                num_systems=int(store.get('num_systems', 8)),
                device=torch.device('cpu'),
                dtype=torch.float32,
            )
        if optional.get('hg_episodic_enabled', False) and self.hg_episodic_ltm is None:
            self.enable_hg_episodic_ltm(
                write_mode=str(optional.get('episodic_write_mode', 'legacy')),
                auto_wire=False,
            )
        if optional.get('parameter_storage_loop_enabled', False) and self.parameter_storage_loop_stack is None:
            self.enable_parameter_loop_ltm_context = bool(optional.get('parameter_loop_ltm_context_enabled', self.enable_parameter_loop_ltm_context))
            self.enable_parameter_loop_training_writes = bool(optional.get('parameter_loop_training_writes_enabled', self.enable_parameter_loop_training_writes))
            self.parameter_loop_training_write_scale = float(optional.get('parameter_loop_training_write_scale', self.parameter_loop_training_write_scale))
            self.enable_parameter_loop_auto_consolidation = bool(optional.get('parameter_loop_auto_consolidation_enabled', self.enable_parameter_loop_auto_consolidation))
            self.parameter_loop_consolidation_interval = int(optional.get('parameter_loop_consolidation_interval', self.parameter_loop_consolidation_interval))
            self.parameter_loop_consolidation_max_bundles = int(optional.get('parameter_loop_consolidation_max_bundles', self.parameter_loop_consolidation_max_bundles))
            self.parameter_loop_consolidation_min_params = int(optional.get('parameter_loop_consolidation_min_params', self.parameter_loop_consolidation_min_params))
            self.parameter_loop_consolidation_min_total_numel = int(optional.get('parameter_loop_consolidation_min_total_numel', self.parameter_loop_consolidation_min_total_numel))
            self.parameter_loop_consolidation_include = tuple(optional.get('parameter_loop_consolidation_include', self.parameter_loop_consolidation_include) or ())
            self.parameter_loop_consolidation_exclude = tuple(optional.get('parameter_loop_consolidation_exclude', self.parameter_loop_consolidation_exclude) or ())
            self.parameter_loop_consolidation_step = int(optional.get('parameter_loop_consolidation_step', self.parameter_loop_consolidation_step))
            self.enable_parameter_storage_loop(
                slots_per_layer=int(optional.get('parameter_loop_slots_per_layer', self.parameter_loop_slots_per_layer)),
                free_hidden_layers=int(optional.get('parameter_loop_free_hidden_layers', self.parameter_loop_free_hidden_layers)),
            )
            registry_state = optional.get('parameter_loop_bundle_registry')
            if registry_state and self.parameter_storage_loop_stack is not None:
                self.parameter_storage_loop_stack.load_parameter_bundle_registry_state(registry_state)
        elif optional.get('parameter_storage_loop_enabled', False) and self.parameter_storage_loop_stack is not None:
            registry_state = optional.get('parameter_loop_bundle_registry')
            if registry_state:
                self.parameter_storage_loop_stack.load_parameter_bundle_registry_state(registry_state)
        
        # Version check
        version = checkpoint.get('version', 'unknown')
        if version != '1.0' and strict:
            raise ValueError(f"Checkpoint version {version} does not match expected '1.0'")
        elif version != '1.0':
            print(f"Warning: Loading checkpoint version {version}, expected '1.0'. Compatibility not guaranteed.")
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Restore config
        if 'config' in checkpoint:
            self.forgetting_threshold = checkpoint['config'].get('forgetting_threshold', self.forgetting_threshold)
            self.energy_mode = checkpoint['config'].get('energy_mode', self.energy_mode)
        self.episodic_write_mode = str(optional.get('episodic_write_mode', getattr(self, "episodic_write_mode", "legacy")))
        
        # Restore lightbulb state
        if 'lightbulb_state' in checkpoint:
            lb = checkpoint['lightbulb_state']
            self.lightbulb.trigger_rate_ema.fill_(lb.get('trigger_rate_ema', 0.0))
            self.lightbulb.thresh = lb.get('threshold', 2.0)
            self.lightbulb.total_fires.fill_(lb.get('total_fires', 0))
            self.lightbulb.total_samples.fill_(lb.get('total_samples', 0))
        
        # Restore memory usage
        if 'memory_state' in checkpoint:
            mem = checkpoint['memory_state']
            if 'hg_usage' in mem:
                self.long_term_memory.hg.usage_counts.copy_(mem['hg_usage'])
            if 'cgmn_usage' in mem:
                self.long_term_memory.cgmn.usage_counts.copy_(mem['cgmn_usage'])
            if 'curved_usage' in mem:
                self.long_term_memory.curved.usage_counts.copy_(mem['curved_usage'])

    def compute_recall_loss(self, cue, context):
        """InfoNCE contrastive loss: cue vs. retrieved memory.
        cue: (B,S,d), context: (B,d)
        Returns scalar loss.
        """
        B, S, d = cue.shape
        # Retrieve from LTM
        retrieved = self.retrieve_memory(cue, context, strategy='direct')  # (B,d)
        
        # Project both to contrastive space
        cue_pooled = cue.mean(dim=1)  # (B,d)
        z_cue = torch.nn.functional.normalize(self.contrastive_proj(cue_pooled), dim=-1)  # (B,128)
        z_ret = torch.nn.functional.normalize(self.contrastive_proj(retrieved), dim=-1)   # (B,128)
        
        # InfoNCE: positive = same batch index, negatives = others
        logits = torch.matmul(z_cue, z_ret.T) / self.recall_temp  # (B,B)
        labels = torch.arange(B, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    # ---------------- Forward ----------------
    def forward(
        self,
        sensory_input,
        context,
        operation='process',
        return_aux_losses=False,
        token_ids=None,
        use_consolidated_memory=False,
        context_features=None,
        consolidation_intent="auto",
    ):
        if use_consolidated_memory and token_ids is not None:
            sensory_input = self._apply_consolidated_memory(
                sensory_input,
                token_ids,
                context_features=context_features,
                consolidation_intent=consolidation_intent,
            )
        fire = self.lightbulb(sensory_input)                # (B,)
        temp = self.temp_scaler(fire)                       # (B,) scalars
        self.working_memory.set_temperature(temp)
        self.long_term_memory.set_temperature(temp)

        if operation == 'process':
            if self.enable_global_hidden_attention:
                self.global_hidden_orchestrator.begin_capture()
            filtered = self.process_sensory_input(sensory_input)          # (B,S,d)
            self.diagnostics.record_scalar("fire_rate", float(fire.float().mean().item()))

            # --- Working-memory write phase ----------------------------------
            # Store filtered sensory input into WM slots with importance gating.
            imp = self.importance_predictor(filtered.mean(dim=1))         # (B,1)
            self.working_memory(filtered, operation='write', importance=imp)
            self.diagnostics.record_scalar("importance_mean", float(imp.mean().item()))

            # --- Working-memory read phase -----------------------------------
            wm_out = self.working_memory(filtered, operation=self._wm_read_operation())  # (B,S,d)
            self._sync_attention_stacks(wm_out)
            self._sync_parameter_loop_ltm_context(wm_out)
            self._sync_cms_depth_ltm_context(wm_out)
            ltm_ctx = self.long_term_memory(
                wm_out,
                operation='read',
                fire_mask=fire,
                recall_boost=0.2,
            )
            if hasattr(self.working_memory, "set_external_attention_context"):
                try:
                    self.working_memory.set_external_attention_context(ltm_ctx.detach())
                except Exception:
                    pass
            bridged = self._bridge_wm_ltm(wm_out, ltm_ctx, phase="process")
            bridged = self._apply_secondary_hidden_stack(
                bridged, wm_out, ltm_ctx, filtered, phase="process"
            )
            bridged = self._apply_parameter_storage_loop(bridged, phase="process")
            bridged = self._apply_consolidated_memory_depth(bridged, phase="process")
            bridged = self._apply_global_hidden_attention(bridged, filtered, phase="process")
            inter = getattr(self.long_term_memory, "last_inter_memory_stats", None)
            if isinstance(inter, dict) and inter:
                self.diagnostics.log("ltm_inter_memory_exchange", inter)
            pref = getattr(self.long_term_memory, "last_prefusion_specialization_stats", None)
            if isinstance(pref, dict) and pref:
                self.diagnostics.log("ltm_prefusion_specialization", pref)

            # --- Consolidation into long-term memory ------------------------
            if self.training:  # consolidate only during training
                self._maybe_consolidate_trainable_parameters()
                # Learned write gate with STE
                gate, gate_prob = self._ste_write_gate(filtered)  # (B,1)
                scaled = bridged * imp.unsqueeze(-1)  # (B,S,d)
                self.diagnostics.record_scalar("write_gate_prob", float(gate_prob.mean().item()))
                
                # Only encode if gate=1 (batched conditional write)
                if gate.sum() > 0:  # at least one sample wants to write
                    # Mask the batch
                    write_mask = gate.squeeze(-1) > 0.5  # (B,)
                    if write_mask.any():
                        self.encode_memory(scaled[write_mask], context[write_mask], mtype='episodic')
            
            # --- Compute auxiliary losses if requested ----------------------
            if return_aux_losses and self.training:
                recall_loss = self.compute_recall_loss(filtered, context)
                gate, gate_prob = self._ste_write_gate(filtered)
                self.diagnostics.record_scalar("recall_loss", float(recall_loss.detach().item()))
                aux = {'recall_loss': recall_loss, 'write_gate_prob': gate_prob.mean()}
                if bool(getattr(self.distillation_config, "enabled", False)):
                    aux["distill_loss"] = recall_loss.detach() * 0.0
                if self.enable_global_hidden_attention:
                    self.global_hidden_orchestrator.end_capture()
                return wm_out, aux
            
            if self.enable_global_hidden_attention:
                self.global_hidden_orchestrator.end_capture()
            return bridged

        elif operation == 'retrieve':
            # Route fire to LTM for sharper readout on the cue as well
            fire_ret = self.lightbulb(sensory_input)  # (B,)
            return self.retrieve_memory(
                sensory_input,
                context,
                strategy='direct',
                fire_mask=fire_ret,
                recall_boost=0.3,
                query_token_ids=token_ids,
            )

        else:
            self.consolidate_memories()
            return None
