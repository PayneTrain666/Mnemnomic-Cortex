import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import warnings
from dataclasses import asdict, is_dataclass
from collections import deque
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
from .config_loader import (
    apply_config_to_broker,
    apply_unified_config_to_cortex,
    load_unified_yaml_config,
    load_yaml_config,
)
from .cps import ConsolidatedParamStore, UnifiedParamCfg
from .cps_fuser import CPSFuser, FuserCfg
from .diagnostics import ModelDiagnostics
from .consolidated_memory import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from .consolidation_broker_v2 import BrokerCfg, ConsolidationBrokerV2
from .consolidation_scheduler import ConsolidationScheduler, SchedCfg
from .multi_cps import MultiCPSManager
from .router_advanced import AdvancedDomainRouter
from .distillation import CrossDomainDistiller, DistillationConfig
from .quantization import CPSQuantizer, QuantPolicy
from .quant_fuser import QuantAwareCPSFuser
from .router_losses import router_regularizer
from .lightbulb_recall_v2 import LightbulbRecallV2

class EnhancedMnemonicCortex(nn.Module):
    """Top-level controller that routes inputs through buffer → WM → LTM with
    lightbulb-triggered 'explosive recall' (temperature modulation).
    Adds:
      • enable_energy_mode()
      • forgetting-style consolidation via consolidate_memories(threshold)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 sensory_buffer_size: int = 5,
                 wm_slots: int = 7, wm_slot_dim: int = 256,
                 ltm_hg_slots: int = 2048, ltm_cgmn_slots: int = 1024, ltm_curved_slots: int = 512,
                 fusion: str = 'weighted',
                 cms_vocab_size: int = 0,
                 cms_senses: int = 3,
                 hgm_enabled: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._ctx_heads = self._pick_num_heads(input_dim)

        self.sensory_buffer = EnhancedSensoryBuffer(sensory_buffer_size, input_dim)
        self.working_memory = EnhancedCurvedMemory(input_dim, hidden_dim=wm_slot_dim, mem_slots=wm_slots)
        self.long_term_memory = EnhancedTripleHybridMemory(input_dim, output_dim,
                                                           hg_slots=ltm_hg_slots, cgmn_slots=ltm_cgmn_slots, curved_slots=ltm_curved_slots,
                                                           fusion=fusion)

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

        # Encoding and retrieval heads
        self.hippocampal_encoder = nn.Sequential(nn.Linear(input_dim*2, 512), nn.ReLU(), nn.Linear(512, 256))
        self.r_proj = nn.Linear(input_dim, 256)
        self.cue_to_input = nn.Linear(256, input_dim)
        self.retrieval = nn.Sequential(nn.Linear(256 + input_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

        # Lightbulb + temperature scaler
        self.lightbulb = LightbulbDetector(input_dim, thresh=2.0)
        self.temp_scaler = ExplosiveRecallScaler(base_temp=1.0, min_temp=0.5, boost=0.3)
        self.recall_controller = LightbulbRecallV2(
            in_dim=4,
            threshold=0.72,
            max_hops=2,
            k_expand_mult=1.6,
        )

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
        self.distillation_config = DistillationConfig(enabled=False)
        self.advanced_quantizers = {}
        self.quant_fuser = None
        self.advanced_scheduler = ConsolidationScheduler(SchedCfg(interval_sec=5, max_merges_per_tick=256))
        self._advanced_pending_merges: List[tuple] = []
        self.reasoning_controller_api = None
        self.advanced_router_feat_proj = None
        self.last_router_decision = None
        self.diagnostics = ModelDiagnostics(enabled=False)
        self.shared_memory_subsystem = None
        self.hg_episodic_ltm = None
        self.episodic_write_mode = "legacy"  # legacy | mirror | shared_only
        self.hgm_enabled = False
        self.hgm_config = None
        self.hgm_last_result = None
        if int(cms_vocab_size) > 0:
            self.enable_consolidated_lexicon(vocab_size=cms_vocab_size, senses=cms_senses)
        if bool(hgm_enabled):
            self.enable_hypergraph_manifold_bridge(enabled=True)

    @classmethod
    def from_yaml(cls, path: str):
        """
        Convenience constructor: build + configure from one unified YAML file.

        Returns:
            tuple[EnhancedMnemonicCortex, UnifiedCortexConfig]
        """
        from .config_loader import build_cortex_from_yaml

        model, unified = build_cortex_from_yaml(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected build_cortex_from_yaml to return {cls.__name__}, got {type(model).__name__}")
        return model, unified

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    # ---------------- Helpers ----------------
    def enable_hypergraph_manifold_bridge(self, enabled: bool = True, *, hgm_config: Any = None):
        if not bool(enabled):
            self.hgm_enabled = False
            self.hgm_last_result = None
            self.diagnostics.log("hgm_bridge_disabled", {"enabled": False})
            return self
        from .hypergraph_manifold import HGMConfig

        if hgm_config is None:
            cfg = HGMConfig()
        elif isinstance(hgm_config, HGMConfig):
            cfg = hgm_config
        elif isinstance(hgm_config, dict):
            cfg = HGMConfig(**dict(hgm_config))
        else:
            raise ValueError("hgm_config must be None, HGMConfig, or dict")
        self.hgm_enabled = True
        self.hgm_config = cfg
        self.diagnostics.log(
            "hgm_bridge_enabled",
            {
                "enabled": True,
                "max_depth_layers": int(cfg.max_depth_layers),
                "max_hyperedge_nodes": int(cfg.max_hyperedge_nodes),
                "default_normalization": str(cfg.default_normalization.value),
            },
        )
        return self

    def run_hypergraph_manifold(
        self,
        mutation_tokens: Any,
        *,
        top_k: int = 8,
        charts: Optional[List[Any]] = None,
        normalization_mode: str = "mutation_axis",
    ) -> Dict[str, Any]:
        if not self.hgm_enabled:
            return {"enabled": False, "reason": "hgm bridge disabled"}
        from .hypergraph_manifold import (
            GeometryType,
            ManifoldChart,
            build_hgm1_scenario_graph,
            build_hgm2_manifold_routing,
            build_probability_expansion,
            extract_top_k_scenarios,
        )

        cfg = self.hgm_config
        expansion = build_probability_expansion(
            mutation_tokens,
            config=cfg,
            normalization_mode=normalization_mode,
        )
        variable_ids = expansion.metadata.get("variable_ids", tuple()) if isinstance(expansion.metadata, dict) else tuple()
        magnitude_ids = expansion.metadata.get("magnitude_bin_ids", tuple()) if isinstance(expansion.metadata, dict) else tuple()
        scenarios = extract_top_k_scenarios(
            expansion.payload,
            k=max(1, int(top_k)),
            contract=expansion.contract,
            config=cfg,
            variable_ids=variable_ids,
            magnitude_bin_ids=magnitude_ids,
        )
        hgm1 = build_hgm1_scenario_graph(scenarios.candidates, config=cfg)
        if charts is None:
            charts = [
                ManifoldChart(chart_id="ctx-euclid", geometry=GeometryType.EUCLIDEAN, dimension=int(self.input_dim)),
                ManifoldChart(chart_id="ctx-hyper", geometry=GeometryType.HYPERBOLIC, dimension=int(self.input_dim)),
                ManifoldChart(chart_id="ctx-product", geometry=GeometryType.PRODUCT, dimension=int(self.input_dim)),
            ]
        hgm2 = build_hgm2_manifold_routing(hgm1.binding.hyperedges, charts=tuple(charts), config=cfg)
        self.hgm_last_result = {
            "expansion": expansion,
            "scenarios": scenarios,
            "hgm1": hgm1,
            "hgm2": hgm2,
        }
        self.diagnostics.log(
            "hgm_pipeline_run",
            {
                "expansion_ok": bool(expansion.validation.ok),
                "scenario_count": int(len(scenarios.candidates)),
                "hyperedge_count": int(len(hgm1.binding.hyperedges)),
                "assignment_count": int(len(hgm2.routing.assignments)),
                "hgm2_ok": bool(hgm2.validation.ok),
            },
        )
        return self.hgm_last_result

    def enable_energy_mode(self, enable: bool = True):
        self.energy_mode = enable
        self.long_term_memory.enable_energy_efficient_mode(enable)
        self.working_memory.enable_energy_efficient_mode(enable)

    def enable_shared_memory_subsystem(
        self,
        *,
        num_slots: int = 2048,
        num_systems: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        geometry_runtime: Any = None,
        reranker: Any = None,
        truth_runtime: Any = None,
        overwrite_threshold: float = 0.35,
        merge_threshold: float = 0.65,
        quarantine_interference_threshold: float = 0.85,
        contradiction_split_threshold: int = 3,
    ):
        """
        Attach the shared-slot memory stack (store/allocator/arbitrator/retention/read/write).
        This is optional and does not alter the main forward path unless used explicitly.
        """
        from .memory import SharedSlotStore, build_shared_memory_subsystem
        from .memory.shared_slot_schema import MEMORY_SYSTEM_IDS

        device = device or self.ctx_proj.weight.device
        dtype = dtype or self.ctx_proj.weight.dtype
        resolved_num_systems = int(len(MEMORY_SYSTEM_IDS) if num_systems is None else num_systems)
        if resolved_num_systems <= 0:
            raise ValueError("num_systems must be positive")
        if resolved_num_systems > len(MEMORY_SYSTEM_IDS):
            raise ValueError(
                f"num_systems ({resolved_num_systems}) cannot exceed schema registry size ({len(MEMORY_SYSTEM_IDS)})"
            )
        store = SharedSlotStore(
            num_slots=int(num_slots),
            slot_dim=int(self.input_dim),
            num_systems=resolved_num_systems,
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
                "num_systems": int(resolved_num_systems),
                "overwrite_threshold": float(overwrite_threshold),
                "merge_threshold": float(merge_threshold),
                "quarantine_interference_threshold": float(quarantine_interference_threshold),
                "contradiction_split_threshold": int(contradiction_split_threshold),
            },
        )
        return self

    def memory_write(self, request: Any, values: torch.Tensor):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        return self.shared_memory_subsystem.write_engine.write(request=request, values=values)

    def memory_read(self, request: Any):
        if self.shared_memory_subsystem is None:
            raise RuntimeError("shared memory subsystem not enabled")
        return self.shared_memory_subsystem.read_engine.retrieve(request)

    def memory_update(self, request: Any):
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
        long_episode_threshold: int = 16,
        summary_stride: int = 8,
        promotion_retrieval_threshold: int = 3,
        write_mode: str = "mirror",
    ):
        """
        Attach HG episodic LTM on top of the shared-memory subsystem.
        If shared memory is not enabled yet, it is enabled with defaults first.
        """
        if self.shared_memory_subsystem is None:
            self.enable_shared_memory_subsystem(
                num_slots=2048,
                num_systems=None,
                device=self.ctx_proj.weight.device,
                dtype=self.ctx_proj.weight.dtype,
            )

        from .ltm.hg_episodic_ltm import HGEpisodicLTM

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
        )
        mode = str(write_mode).lower().strip()
        if mode not in {"legacy", "mirror", "shared_only"}:
            raise ValueError(f"unsupported write_mode '{write_mode}'")
        self.episodic_write_mode = mode
        self.diagnostics.log(
            "hg_episodic_ltm_enabled",
            {
                "long_episode_threshold": int(long_episode_threshold),
                "summary_stride": int(summary_stride),
                "promotion_retrieval_threshold": int(promotion_retrieval_threshold),
                "write_mode": mode,
            },
        )
        return self

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
        return self.hg_episodic_ltm.store_episode(
            episode_id=episode_id,
            episode_vectors=episode_vectors,
            step_range=step_range,
            anchor_time=anchor_time,
            trace_ids=trace_ids,
            tags=tags,
        )

    def retrieve_episodic_trace(
        self,
        *,
        query: torch.Tensor,
        top_k: int = 16,
        tags: Optional[List[str]] = None,
        time_window=None,
    ):
        if self.hg_episodic_ltm is None:
            raise RuntimeError("hg episodic ltm not enabled")
        return self.hg_episodic_ltm.retrieve_episode_fragments(
            query=query,
            top_k=top_k,
            tags=tags,
            time_window=time_window,
        )

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
            self.configure_distillation(gcfg.distill)
        except FileNotFoundError:
            warnings.warn(
                f"Broker config '{config_path}' not found; using default broker/AHG settings.",
                RuntimeWarning,
            )
        except RuntimeError as exc:
            warnings.warn(f"Broker config load failed: {exc}", RuntimeWarning)
        except Exception as exc:
            warnings.warn(f"Broker config apply failed: {exc}", RuntimeWarning)
        return self

    def configure_from_yaml(
        self,
        path: str,
        *,
        auto_enable_broker: bool = True,
        broker_vocab_size: Optional[int] = None,
    ):
        unified = load_unified_yaml_config(path)
        ctor_cfg = unified.cortex

        if int(ctor_cfg.input_dim) != int(self.input_dim) or int(ctor_cfg.output_dim) != int(self.output_dim):
            warnings.warn(
                "YAML cortex dims differ from existing model instance "
                f"(yaml: in={int(ctor_cfg.input_dim)}, out={int(ctor_cfg.output_dim)}; "
                f"model: in={int(self.input_dim)}, out={int(self.output_dim)}). "
                "Keeping existing instantiated dimensions.",
                RuntimeWarning,
            )

        if auto_enable_broker and self.consolidation_broker is None and unified.global_config.stores:
            resolved_vocab = int(
                broker_vocab_size
                or getattr(self.consolidated_lexicon, "vocab_size", 0)
                or int(getattr(ctor_cfg, "cms_vocab_size", 0))
            )
            if resolved_vocab > 0:
                if self.consolidated_lexicon is None and int(getattr(ctor_cfg, "cms_vocab_size", 0)) > 0:
                    self.enable_consolidated_lexicon(
                        vocab_size=int(ctor_cfg.cms_vocab_size),
                        senses=int(ctor_cfg.cms_senses),
                    )
                self.enable_consolidation_broker(vocab_size=resolved_vocab, config_path=path)
            else:
                warnings.warn(
                    "Unified YAML requests broker/store config, but vocab size is unavailable. "
                    "Provide broker_vocab_size or set cortex.cms_vocab_size in YAML.",
                    RuntimeWarning,
                )

        apply_unified_config_to_cortex(self, unified)
        self.diagnostics.log(
            "configured_from_yaml",
            {
                "path": str(path),
                "hgm_enabled": bool(self.hgm_enabled),
                "reasoning_bridge_enabled": bool(self.reasoning_controller_api is not None),
                "qdt_wm_bridge_enabled": bool(self._resolve_qdt_working_memory() is not None),
                "shared_memory_enabled": bool(self.shared_memory_subsystem is not None),
                "hg_episodic_ltm_enabled": bool(self.hg_episodic_ltm is not None),
                "broker_enabled": bool(self.consolidation_broker is not None),
            },
        )
        return unified

    def configure_distillation(self, cfg: Any = None):
        if cfg is None:
            cfg = DistillationConfig(enabled=False)
        elif isinstance(cfg, dict):
            cfg = DistillationConfig(**cfg)
        elif not isinstance(cfg, DistillationConfig):
            raise ValueError("cfg must be DistillationConfig, dict, or None")
        cfg.validate()
        self.distillation_config = cfg
        if self.advanced_distiller is not None:
            self.advanced_distiller.configure(cfg)
        self.diagnostics.log(
            "distillation_configured",
            {
                "enabled": bool(cfg.enabled),
                "teacher_domain": str(cfg.teacher_domain),
                "student_domains": [str(x) for x in cfg.student_domains],
                "neighbor_k": int(cfg.neighbor_k),
                "sim_temp": float(cfg.sim_temp),
            },
        )
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
        cms_cfg = cms_cfg or ConsolidatedMemoryCfg(d_model=self.input_dim)
        broker_cfg = broker_cfg or BrokerCfg()
        domains = list(cps_domains or ["core", "science", "reasoning", "creativity"])

        self.advanced_cms = ConsolidatedMemoryStore(cms_cfg)
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
        self.advanced_distiller.configure(self.distillation_config)
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
        self.diagnostics.log(
            "advanced_consolidation_enabled",
            {
                "domains": list(self.multi_cps.cps.keys()),
                "distillation_enabled": bool(self.distillation_config.enabled),
            },
        )
        self._wire_runtime_external_memory_backends()
        return self

    @staticmethod
    def _distill_keys_from_token_ids(token_ids) -> List[str]:
        if token_ids is None:
            return []
        if token_ids.dim() > 1:
            flat = token_ids.reshape(-1)
        else:
            flat = token_ids
        seen = set()
        out = []
        for t in flat.tolist():
            key = f"token:{int(t)}"
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def enable_reasoning_controller_bridge(
        self,
        *,
        enabled: bool = True,
        allow_shared_mann_ltm_geometry: bool = True,
        max_reasoning_hops: int = 2,
    ):
        from .reasoning_depth.reasoning_controller_api import (
            ReasoningControllerAPI,
            ReasoningControllerAPIConfig,
        )

        if not enabled:
            self.reasoning_controller_api = None
            self.diagnostics.log("reasoning_controller_bridge", {"enabled": False})
            return self

        cfg = ReasoningControllerAPIConfig(
            enabled=True,
            key_dim=int(self.input_dim),
            value_dim=int(self.input_dim),
            slot_count=32,
            max_reasoning_hops=int(max(1, max_reasoning_hops)),
            allow_shared_mann_ltm_geometry=bool(allow_shared_mann_ltm_geometry),
            allow_policy_router=True,
            allow_evidence_reasoning=True,
            allow_counterfactual_probe=True,
            allow_conflict_aware_consolidation=True,
            require_json_safe_outputs=False,
        )
        self.reasoning_controller_api = ReasoningControllerAPI(cfg)
        self.diagnostics.log(
            "reasoning_controller_bridge",
            {
                "enabled": True,
                "shared_geometry": bool(allow_shared_mann_ltm_geometry),
                "max_reasoning_hops": int(max_reasoning_hops),
            },
        )
        return self

    def enable_qdt_working_memory_bridge(
        self,
        *,
        hidden_dim: int = 128,
        num_depths: int = 8,
        num_slots: int = 8,
        num_heads: Optional[int] = None,
        transformer_layers: int = 1,
        use_compatibility_wrapper: bool = True,
    ):
        from .working_memory import CortexWorkingMemoryIntegrationConfig, replace_cortex_working_memory

        resolved_heads = int(self._pick_num_heads(int(self.input_dim)) if num_heads is None else num_heads)
        cfg = CortexWorkingMemoryIntegrationConfig(
            input_dim=int(self.input_dim),
            hidden_dim=int(hidden_dim),
            num_depths=int(num_depths),
            num_slots=int(num_slots),
            num_heads=resolved_heads,
            transformer_layers=int(transformer_layers),
            use_compatibility_wrapper=bool(use_compatibility_wrapper),
            preserve_old_reference=True,
        )
        result = replace_cortex_working_memory(self, cfg)
        self.diagnostics.log("qdt_working_memory_bridge", result.to_dict())
        self._wire_runtime_external_memory_backends()
        return self

    def enable_cps_cms_full_stack(
        self,
        *,
        vocab_size: int,
        cms_senses: int = 3,
        enable_broker: bool = True,
        enable_advanced: bool = True,
        enable_reasoning_bridge: bool = True,
        enable_qdt_wm_bridge: bool = True,
    ):
        self.enable_consolidated_lexicon(vocab_size=int(vocab_size), senses=int(cms_senses))
        if enable_broker:
            self.enable_consolidation_broker(vocab_size=int(vocab_size))
        if enable_advanced:
            self.enable_advanced_consolidation()
        if enable_reasoning_bridge:
            self.enable_reasoning_controller_bridge(enabled=True, allow_shared_mann_ltm_geometry=True)
        if enable_qdt_wm_bridge:
            self.enable_qdt_working_memory_bridge()
        else:
            self._wire_runtime_external_memory_backends()
        self.diagnostics.log(
            "cps_cms_full_stack_enabled",
            {
                "vocab_size": int(vocab_size),
                "cms_senses": int(cms_senses),
                "broker": bool(enable_broker),
                "advanced": bool(enable_advanced),
                "reasoning_bridge": bool(enable_reasoning_bridge),
                "qdt_wm_bridge": bool(enable_qdt_wm_bridge),
            },
        )
        return self

    def _resolve_qdt_working_memory(self):
        wm = self.working_memory
        if hasattr(wm, "qdt_working_memory"):
            return getattr(wm, "qdt_working_memory")
        if hasattr(wm, "dual_fusion"):
            return wm
        return None

    def _wire_runtime_external_memory_backends(self) -> bool:
        from .working_memory.wm_external_memory_interfaces import RuntimeExternalMemoryBank

        qdt_wm = self._resolve_qdt_working_memory()
        if qdt_wm is None or not hasattr(qdt_wm, "dual_fusion"):
            return False

        shared_slot_store = getattr(qdt_wm, "shared_slot_store", None)

        def _ltm_query_fn(request, top_k: int):
            query = request.query_state
            qseq = query.unsqueeze(1)
            values = self.long_term_memory(qseq, operation="read")
            pooled = values.mean(dim=1)
            k = max(1, int(top_k))
            memory_state = pooled.unsqueeze(1).expand(-1, k, -1).contiguous()
            qn = torch.nn.functional.normalize(query, dim=-1)
            mn = torch.nn.functional.normalize(memory_state[:, 0, :], dim=-1)
            score = (qn * mn).sum(dim=-1, keepdim=True)
            scores = score.expand(-1, k).contiguous()
            slot_ids = [[f"ltm_runtime_{i}" for i in range(k)] for _ in range(query.size(0))]
            return {"memory_state": memory_state, "scores": scores, "slot_ids": slot_ids}

        def _mann_query_fn(request, top_k: int):
            query = request.query_state
            if self.reasoning_controller_api is not None:
                try:
                    api_result = self.reasoning_controller_api.run_reasoning_pass(
                        query,
                        content="wm_mann_runtime_bridge",
                        write_permission=False,
                    )
                    out = api_result.result.mann_output
                    if out is None:
                        out = api_result.result.output
                    mann_state = out
                except Exception:
                    mann_state = query
            else:
                mann_state = query
            k = max(1, int(top_k))
            memory_state = mann_state.unsqueeze(1).expand(-1, k, -1).contiguous()
            qn = torch.nn.functional.normalize(query, dim=-1)
            mn = torch.nn.functional.normalize(memory_state[:, 0, :], dim=-1)
            score = (qn * mn).sum(dim=-1, keepdim=True)
            scores = score.expand(-1, k).contiguous()
            hops = 3
            hop_scales = torch.linspace(0.25, 1.0, steps=hops, device=query.device, dtype=query.dtype).view(1, hops, 1)
            scratchpad_tokens = mann_state.unsqueeze(1) * hop_scales
            per_hop_attention = torch.softmax(scores.unsqueeze(1).expand(-1, hops, -1), dim=-1)
            slot_ids = [[f"mann_runtime_{i}" for i in range(k)] for _ in range(query.size(0))]
            return {
                "memory_state": memory_state,
                "scores": scores,
                "slot_ids": slot_ids,
                "scratchpad_tokens": scratchpad_tokens,
                "per_hop_attention": per_hop_attention,
            }

        def _spcp_query_fn(request, top_k: int):
            query = request.query_state
            qseq = query.unsqueeze(1)
            values = self.long_term_memory.curved(qseq, operation="read")
            pooled = values.mean(dim=1)
            k = max(1, int(top_k))
            memory_state = pooled.unsqueeze(1).expand(-1, k, -1).contiguous()
            qn = torch.nn.functional.normalize(query, dim=-1)
            mn = torch.nn.functional.normalize(memory_state[:, 0, :], dim=-1)
            score = (qn * mn).sum(dim=-1, keepdim=True)
            scores = score.expand(-1, k).contiguous()
            slot_ids = [[f"spcp_runtime_{i}" for i in range(k)] for _ in range(query.size(0))]
            return {"memory_state": memory_state, "scores": scores, "slot_ids": slot_ids}

        qdt_wm.dual_fusion.ltm.external_bank = RuntimeExternalMemoryBank(
            memory_type="ltm",
            dim=int(self.input_dim),
            query_fn=_ltm_query_fn,
            shared_slot_store=shared_slot_store,
        )
        qdt_wm.dual_fusion.mann.external_bank = RuntimeExternalMemoryBank(
            memory_type="mann",
            dim=int(self.input_dim),
            query_fn=_mann_query_fn,
            shared_slot_store=shared_slot_store,
        )
        qdt_wm.dual_fusion.spcp.external_bank = RuntimeExternalMemoryBank(
            memory_type="spcp",
            dim=int(self.input_dim),
            query_fn=_spcp_query_fn,
            shared_slot_store=shared_slot_store,
        )
        self.diagnostics.log(
            "wm_external_runtime_backends_wired",
            {"ltm": "runtime", "mann": "runtime", "spcp": "runtime"},
        )
        return True

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

    @torch.no_grad()
    def _enqueue_advanced_ltm_merge(
        self,
        *,
        key: str,
        candidate_view: Dict[str, torch.Tensor],
        importance: torch.Tensor,
        src_info: Dict[str, Any],
    ) -> None:
        if self.advanced_broker is None:
            return
        self._advanced_pending_merges.append((key, candidate_view, importance, src_info))
        if len(self._advanced_pending_merges) > 4096:
            self._advanced_pending_merges = self._advanced_pending_merges[-4096:]

    @torch.no_grad()
    def _tick_advanced_consolidation(self, token_keys: Optional[List[str]] = None) -> None:
        if self.advanced_broker is None:
            return

        token_keys = token_keys or []
        # Preserve insertion order while deduping.
        seen = set()
        deduped = []
        for k in token_keys:
            if k in seen:
                continue
            seen.add(k)
            deduped.append(k)

        def _pending_merges():
            pending = list(self._advanced_pending_merges)
            self._advanced_pending_merges.clear()
            return pending

        def _nudge_keys():
            if deduped:
                return deduped[:256]
            return self.cps.keys()[:256]

        def _reindex_keys():
            return []

        self.advanced_scheduler.tick(
            pending_merges=_pending_merges,
            nudge_keys=_nudge_keys,
            reindex_keys=_reindex_keys,
            broker=self.advanced_broker,
            cms_index=None,
        )

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
        self.last_cms_aux = None
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
            if self.consolidated_lexicon is not None:
                fused, _, aux = self.consolidated_lexicon(flat_ids, flat_base, flat_ctx)
                self.last_cms_aux = aux
                self.diagnostics.log("cms_single_store", {"intent": consolidation_intent})
            else:
                # CPS can still provide token-level consolidation without CMS banks.
                fused = flat_base
                aux = {}
                self.diagnostics.log("cms_unavailable_cps_only", {"intent": consolidation_intent})

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
        quantized_fusion_count = 0
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
                dom = "core"
            if self.quant_fuser is not None and isinstance(self.advanced_quantizers, dict):
                quantizer = self.advanced_quantizers.get(dom) or self.advanced_quantizers.get("core")
                if quantizer is not None:
                    try:
                        self.quant_fuser.quantizer = quantizer
                        qpack = quantizer.quantize_entry(up)
                        qv, _ = self.quant_fuser(qpack=qpack, device=fused.device)
                        v = 0.5 * v + 0.5 * qv.to(v.device, v.dtype)
                        quantized_fusion_count += 1
                    except Exception as exc:
                        self.diagnostics.log("quant_fuser_error", {"error": str(exc), "domain": str(dom)})
            cps_fused.append(v)
            cps_loss = cps_loss + loss
        if cps_fused:
            cps_fused = torch.stack(cps_fused, dim=0).to(fused.device, fused.dtype)
            fused = 0.85 * fused + 0.15 * cps_fused
            cps_loss = cps_loss / max(1, len(cps_fused))
            self.last_cps_aux = {"agree_loss": cps_loss}
            self.last_cps_aux["quantized_fusion_count"] = int(quantized_fusion_count)
            token_keys = [f"token:{int(t)}" for t in flat_ids.tolist()]
            if self.advanced_broker is not None:
                try:
                    self._tick_advanced_consolidation(token_keys=token_keys)
                    coh = self.advanced_broker.cohesion_regularizer(token_keys[:128])
                    self.last_cps_aux["advanced_cohesion"] = coh.detach()
                    self.diagnostics.record_scalar("advanced_cms_cohesion", float(coh.detach().item()))
                except Exception as exc:
                    self.diagnostics.log("advanced_consolidation_tick_error", {"error": str(exc)})
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
        ctx = self._tile_context(context, S)
        pooled = torch.cat([info, ctx], dim=-1).mean(dim=1)      # (B, 2d)
        idx = self.hippocampal_encoder(pooled)                   # (B,256)

        cue = idx.unsqueeze(1).expand(-1, S, -1)                 # (B,S,256)
        # Project cue to input_dim if needed
        if cue.size(-1) != self.input_dim:
            cue = self.cue_to_input(cue)

        if mtype == 'episodic':
            if self.episodic_write_mode in {"legacy", "mirror"}:
                self.long_term_memory.hg(cue, operation='write')
            if self.hg_episodic_ltm is not None and self.episodic_write_mode in {"mirror", "shared_only"}:
                for b in range(B):
                    self.store_episodic_trace(
                        episode_id=f"auto-ep-{int(self.shared_memory_subsystem.store.version_counter if self.shared_memory_subsystem is not None else 0)}-{int(b)}",
                        episode_vectors=info[b],
                        step_range=(0, int(S - 1)),
                        trace_ids=[f"cortex:auto:{int(b)}"],
                        tags=["auto", "episodic", "cortex"],
                    )
        elif mtype == 'semantic':
            self.long_term_memory.cgmn(cue, operation='write')
        else:
            self.long_term_memory.curved(cue, operation='write')

        if self.advanced_broker is not None:
            for b in range(B):
                cvec = cue[b].mean(dim=0).detach()
                importance = torch.sigmoid(cvec.norm().view(1))
                self._enqueue_advanced_ltm_merge(
                    key=f"ltm:{str(mtype)}:b{int(b)}",
                    candidate_view={"E": cvec},
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
        B,S,d = cue.shape
        ctx = self._tile_context(context, S)
        c = (cue + ctx) * 0.5
        qctx, _ = self.query_ctx_attn(c, ctx, ctx, need_weights=False)
        c = self.query_norm(c + qctx)
        if fire_mask is None:
            fire_mask = self.lightbulb(cue)
        if isinstance(fire_mask, torch.Tensor):
            fire_rate = float(fire_mask.float().mean().item())
            has_fire = bool(fire_mask.any().item())
        else:
            has_fire = bool(fire_mask)
            fire_rate = 1.0 if has_fire else 0.0
        recall_signal = {
            "entropy_drop": 0.0,
            "agreement": 0.5,
            "novelty": 0.0,
            "uncertainty": 0.0,
        }

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
                return self.retrieval(torch.cat([z, context], dim=-1))
            recall_signal["agreement"] = float(decision.scores.get("agree", 0.5))
            recall_signal["uncertainty"] = float(decision.scores.get("fisher", 0.0))

        cue_p = torch.softmax(cue.detach().abs(), dim=-1)
        ctx_p = torch.softmax(ctx.detach().abs(), dim=-1)
        cue_entropy = -(cue_p * cue_p.clamp_min(1e-9).log()).sum(dim=-1).mean()
        ctx_entropy = -(ctx_p * ctx_p.clamp_min(1e-9).log()).sum(dim=-1).mean()
        recall_signal["entropy_drop"] = float((ctx_entropy - cue_entropy).detach().clamp_min(0.0).item())
        novelty = (cue.detach() - ctx.detach()).norm(dim=-1).mean()
        recall_signal["novelty"] = float(torch.sigmoid(novelty / max(1.0, float(cue.size(-1) ** 0.5))).item())
        recall_event = self.recall_controller(
            entropy_drop=torch.tensor(recall_signal["entropy_drop"], device=cue.device),
            cms_cps_agreement=torch.tensor(recall_signal["agreement"], device=cue.device),
            novelty=torch.tensor(recall_signal["novelty"], device=cue.device),
            uncertainty=torch.tensor(recall_signal["uncertainty"], device=cue.device),
            base_k=int(getattr(self.long_term_memory, "K_base", 8)),
            debounce_ok=not has_fire,
        )
        if recall_event.triggered:
            recall_boost = max(
                recall_boost,
                min(
                    1.25,
                    0.65
                    + 0.10 * max(0, int(recall_event.hops))
                    + 0.05 * max(0.0, float(recall_event.expanded_k) / max(1.0, float(getattr(self.long_term_memory, "K_base", 8))) - 1.0),
                ),
            )
            fire_mask = torch.ones(B, device=cue.device, dtype=torch.bool)
            has_fire = True
            fire_rate = 1.0

        if strategy == 'direct':
            r = self.long_term_memory(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)
        elif strategy == 'associative':
            r = self.long_term_memory.curved(
                c,
                operation='read',
                fire_mask=fire_mask,
                recall_boost=recall_boost,
            )
        else:
            # Reconstructive via HG
            r = self.long_term_memory.hg(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)

        # Bidirectional bridge: retrieved LTM context exchanges with WM readout.
        wm_r = self.working_memory(c, operation='read')
        r = self._bridge_wm_ltm(wm_r, r, phase="retrieve")
        if self.reasoning_controller_api is not None:
            try:
                rc_out = self.reasoning_controller_api.run_reasoning_pass(
                    r.mean(dim=1),
                    content="cortex_retrieve_mann_bridge",
                    write_permission=False,
                )
                mann = rc_out.result.mann_output
                if mann is None:
                    mann = rc_out.result.output
                mann_seq = mann.unsqueeze(1).expand(-1, r.size(1), -1)
                mann_weight = min(0.45, max(0.10, 0.15 + 0.15 * float(recall_event.triggered) + 0.10 * float(fire_rate)))
                align = torch.nn.functional.cosine_similarity(r.mean(dim=1), mann, dim=-1).mean()
                if float(align.item()) < 0.0:
                    mann_weight *= 0.5
                r = (1.0 - mann_weight) * r + mann_weight * mann_seq
                self.diagnostics.log(
                    "reasoning_mann_bridge",
                    {"enabled": True, "mann_weight": float(mann_weight), "alignment": float(align.item())},
                )
            except Exception as exc:
                self.diagnostics.log("reasoning_mann_bridge_error", {"error": str(exc)})
        inter = getattr(self.long_term_memory, "last_inter_memory_stats", None)
        if isinstance(inter, dict) and inter:
            self.diagnostics.log("ltm_inter_memory_exchange", inter)

        cue_vec = self.r_proj(r.mean(dim=1))
        self.diagnostics.record_scalar("recall_boost", float(recall_boost))
        self.diagnostics.record_scalar("recall_fire_rate", float(fire_rate))
        self.diagnostics.log(
            "recall_event",
            {
                "triggered": bool(recall_event.triggered),
                "spike_score": float(recall_event.spike_score),
                "expanded_k": int(recall_event.expanded_k),
                "hops": int(recall_event.hops),
                "signals": recall_signal,
            },
        )
        self.diagnostics.log("retrieve_path", {"strategy": strategy})
        return self.retrieval(torch.cat([cue_vec, context], dim=-1))

    @torch.no_grad()
    def consolidate_memories(self, threshold: float = None):
        """Forget rarely used slots (usage-based) and gently decay working-memory importance."""
        th = self.forgetting_threshold if threshold is None else float(threshold)
        self.long_term_memory.consolidate_unused(th)
        if hasattr(self.working_memory, "memory_importance"):
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
        if self.advanced_broker is not None:
            try:
                self._tick_advanced_consolidation()
                self.diagnostics.log(
                    "advanced_consolidation_tick",
                    {"cps_keys": int(len(self.cps.keys()))},
                )
            except Exception as exc:
                self.diagnostics.log("advanced_consolidation_tick_error", {"error": str(exc)})

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
        qh_banks = getattr(self.long_term_memory, "qh_banks", None)
        if isinstance(qh_banks, nn.ModuleDict):
            for bank_name, bank in qh_banks.items():
                if hasattr(bank, "trace_summary"):
                    for k, v in bank.trace_summary().items():
                        metrics[f"ltm_qh_{bank_name}_{k}"] = float(v)
        wm_qh = getattr(self.working_memory, "qh_slot_bank", None)
        if wm_qh is not None and hasattr(wm_qh, "trace_summary"):
            for k, v in wm_qh.trace_summary().items():
                metrics[f"wm_qh_{k}"] = float(v)
        cms = getattr(self, "advanced_cms", None)
        cms_qh = getattr(cms, "qh_slot_bank", None) if cms is not None else None
        if cms_qh is not None and hasattr(cms_qh, "trace_summary"):
            for k, v in cms_qh.trace_summary().items():
                metrics[f"cms_qh_{k}"] = float(v)
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
        else:
            metrics["hg_episodic_enabled"] = 0.0
        metrics["advanced_cms_enabled"] = 1.0 if self.advanced_cms is not None else 0.0
        metrics["advanced_broker_enabled"] = 1.0 if self.advanced_broker is not None else 0.0
        if self.advanced_cms is not None:
            metrics["advanced_cms_keys"] = float(len(self.advanced_cms.keys()))
        metrics["reasoning_bridge_enabled"] = 1.0 if self.reasoning_controller_api is not None else 0.0
        metrics["hgm_enabled"] = 1.0 if self.hgm_enabled else 0.0
        if isinstance(self.hgm_last_result, dict):
            hgm2 = self.hgm_last_result.get("hgm2", None)
            scenarios = self.hgm_last_result.get("scenarios", None)
            hgm1 = self.hgm_last_result.get("hgm1", None)
            if scenarios is not None:
                metrics["hgm_scenarios"] = float(len(getattr(scenarios, "candidates", tuple())))
            if hgm1 is not None:
                binding = getattr(hgm1, "binding", None)
                if binding is not None:
                    metrics["hgm_hyperedges"] = float(len(getattr(binding, "hyperedges", tuple())))
            if hgm2 is not None:
                routing = getattr(hgm2, "routing", None)
                if routing is not None:
                    metrics["hgm_assignments"] = float(len(getattr(routing, "assignments", tuple())))
                validation = getattr(hgm2, "validation", None)
                metrics["hgm_validation_ok"] = 1.0 if bool(getattr(validation, "ok", False)) else 0.0
        qdt_wm = self._resolve_qdt_working_memory()
        metrics["qdt_wm_bridge_enabled"] = 1.0 if qdt_wm is not None else 0.0
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
        }
        if self.shared_memory_subsystem is not None:
            store = self.shared_memory_subsystem.store
            ser_meta = {}
            for k, v in store.metadata.items():
                if is_dataclass(v):
                    ser_meta[int(k)] = asdict(v)
                elif isinstance(v, dict):
                    ser_meta[int(k)] = dict(v)
                else:
                    ser_meta[int(k)] = v
            checkpoint["shared_memory_state"] = {
                "num_slots": int(store.num_slots),
                "slot_dim": int(store.slot_dim),
                "num_systems": int(store.num_systems),
                "version_counter": int(store.version_counter),
                "free_slot_ids": list(store.free_slot_ids),
                "slot_values": store.slot_values.detach().cpu(),
                "slot_confidence": store.slot_confidence.detach().cpu(),
                "slot_usage": store.slot_usage.detach().cpu(),
                "slot_age": store.slot_age.detach().cpu(),
                "slot_state_code": store.slot_state_code.detach().cpu(),
                "primary_system_code": store.primary_system_code.detach().cpu(),
                "allowed_read_mask": store.allowed_read_mask.detach().cpu(),
                "allowed_write_mask": store.allowed_write_mask.detach().cpu(),
                "metadata": ser_meta,
            }
        if self.hg_episodic_ltm is not None:
            ep = self.hg_episodic_ltm
            checkpoint["hg_episodic_state"] = {
                "episodic_write_mode": str(self.episodic_write_mode),
                "episode_records": {
                    eid: (asdict(rec) if is_dataclass(rec) else dict(rec))
                    for eid, rec in ep.episode_records.items()
                },
                "episode_index_by_trace_id": dict(ep.episode_index_by_trace_id),
                "episode_index_by_time": list(ep.episode_index_by_time),
                "episode_index_by_tag": dict(ep.episode_index_by_tag),
                "slot_retrieval_hits": dict(ep.slot_retrieval_hits),
            }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load checkpoint with version validation."""
        import torch
        from .memory.shared_slot_schema import SlotMetadata, SlotProvenance
        from .ltm.hg_episodic_ltm import EpisodeRecord
        checkpoint = torch.load(path, map_location='cpu')
        
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
        if "shared_memory_state" in checkpoint:
            sm = checkpoint["shared_memory_state"]
            if self.shared_memory_subsystem is None:
                self.enable_shared_memory_subsystem(
                    num_slots=int(sm.get("num_slots", 2048)),
                    num_systems=int(sm.get("num_systems", 8)),
                    device=self.ctx_proj.weight.device,
                    dtype=self.ctx_proj.weight.dtype,
                )
            store = self.shared_memory_subsystem.store
            store.slot_values.copy_(sm["slot_values"].to(device=store.slot_values.device, dtype=store.slot_values.dtype))
            store.slot_confidence.copy_(sm["slot_confidence"].to(device=store.slot_confidence.device, dtype=store.slot_confidence.dtype))
            store.slot_usage.copy_(sm["slot_usage"].to(device=store.slot_usage.device, dtype=store.slot_usage.dtype))
            store.slot_age.copy_(sm["slot_age"].to(device=store.slot_age.device, dtype=store.slot_age.dtype))
            store.slot_state_code.copy_(sm["slot_state_code"].to(device=store.slot_state_code.device, dtype=store.slot_state_code.dtype))
            store.primary_system_code.copy_(sm["primary_system_code"].to(device=store.primary_system_code.device, dtype=store.primary_system_code.dtype))
            store.allowed_read_mask.copy_(sm["allowed_read_mask"].to(device=store.allowed_read_mask.device, dtype=store.allowed_read_mask.dtype))
            store.allowed_write_mask.copy_(sm["allowed_write_mask"].to(device=store.allowed_write_mask.device, dtype=store.allowed_write_mask.dtype))
            store.version_counter = int(sm.get("version_counter", 0))
            store.free_slot_ids = deque(int(x) for x in sm.get("free_slot_ids", []))
            restored_meta = {}
            for key, val in (sm.get("metadata") or {}).items():
                sid = int(key)
                if isinstance(val, dict) and {"slot_id", "state", "confidence", "usage_score", "age_steps", "primary_system_id"}.issubset(val.keys()):
                    prov = val.get("provenance")
                    if isinstance(prov, dict):
                        val["provenance"] = SlotProvenance(**prov)
                    restored_meta[sid] = SlotMetadata(**val)
                else:
                    restored_meta[sid] = val
            store.metadata = restored_meta
        if "hg_episodic_state" in checkpoint:
            hs = checkpoint["hg_episodic_state"]
            if self.hg_episodic_ltm is None:
                self.enable_hg_episodic_ltm(write_mode=str(hs.get("episodic_write_mode", "mirror")))
            else:
                self.episodic_write_mode = str(hs.get("episodic_write_mode", self.episodic_write_mode))
            ep = self.hg_episodic_ltm
            recs = {}
            for eid, raw in (hs.get("episode_records") or {}).items():
                if isinstance(raw, dict):
                    recs[eid] = EpisodeRecord(**raw)
            ep.episode_records = recs
            ep.episode_index_by_trace_id = {k: list(v) for k, v in (hs.get("episode_index_by_trace_id") or {}).items()}
            ep.episode_index_by_time = [tuple(x) for x in (hs.get("episode_index_by_time") or [])]
            ep.episode_index_by_tag = {k: list(v) for k, v in (hs.get("episode_index_by_tag") or {}).items()}
            ep.slot_retrieval_hits = {int(k): int(v) for k, v in (hs.get("slot_retrieval_hits") or {}).items()}

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
        if hasattr(self.working_memory, "set_temperature"):
            self.working_memory.set_temperature(temp)
        self.long_term_memory.set_temperature(temp)

        if operation == 'process':
            filtered = self.process_sensory_input(sensory_input)          # (B,S,d)
            self.diagnostics.record_scalar("fire_rate", float(fire.float().mean().item()))

            # --- Working-memory write phase ----------------------------------
            # Store filtered sensory input into WM slots with importance gating.
            imp = self.importance_predictor(filtered.mean(dim=1))         # (B,1)
            self.working_memory(filtered, operation='write', importance=imp)
            self.diagnostics.record_scalar("importance_mean", float(imp.mean().item()))

            # --- Working-memory read phase -----------------------------------
            wm_out = self.working_memory(filtered, operation='read')      # (B,S,d)
            ltm_ctx = self.long_term_memory(
                wm_out,
                operation='read',
                fire_mask=fire,
                recall_boost=0.2,
            )
            bridged = self._bridge_wm_ltm(wm_out, ltm_ctx, phase="process")
            inter = getattr(self.long_term_memory, "last_inter_memory_stats", None)
            if isinstance(inter, dict) and inter:
                self.diagnostics.log("ltm_inter_memory_exchange", inter)

            # --- Consolidation into long-term memory ------------------------
            if self.training:  # consolidate only during training
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
                distill_loss = torch.tensor(0.0, device=filtered.device, dtype=filtered.dtype)
                if (
                    self.distillation_config.enabled
                    and self.advanced_distiller is not None
                    and self.multi_cps is not None
                    and token_ids is not None
                ):
                    dkeys = self._distill_keys_from_token_ids(token_ids)
                    if dkeys:
                        distill_domains = {
                            str(self.distillation_config.teacher_domain),
                            *[str(d) for d in self.distillation_config.student_domains],
                        }
                        for dom in distill_domains:
                            if self.multi_cps.has_domain(dom):
                                for k in dkeys:
                                    self.multi_cps.ensure(k, domain=dom, device=filtered.device, dtype=filtered.dtype)
                        distill_loss = self.advanced_distiller.total_distill_loss(
                            dkeys,
                            self.distillation_config,
                        )
                self.diagnostics.record_scalar("distill_loss", float(distill_loss.detach().item()))
                return wm_out, {
                    'recall_loss': recall_loss,
                    'distill_loss': distill_loss,
                    'aux_total': recall_loss + distill_loss,
                    'write_gate_prob': gate_prob.mean(),
                }
            
            return bridged

        elif operation == 'retrieve':
            # Route fire to LTM for sharper readout on the cue as well
            return self.retrieve_memory(
                sensory_input,
                context,
                strategy='direct',
                fire_mask=fire,
                recall_boost=0.3,
                query_token_ids=token_ids,
            )

        else:
            self.consolidate_memories()
            return None
