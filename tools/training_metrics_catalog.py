"""
Plain-language summary
----------------------
What this file is for: Dictionary and advice rules for copy/reverse training metrics.
How it fits in the system: Shared by the metrics guide, live dashboard, and trainers.
Status: WORKING
Important notes for non-coders: Explains what each number means and when it looks healthy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


MetricDef = Dict[str, Any]


CORE_TERMINAL_METRICS: Dict[str, MetricDef] = {
    "loss": {
        "label": "Step loss",
        "group": "core",
        "direction": "lower",
        "unit": "nats/token-ish CE",
        "meaning": "Cross-entropy on the current batch of reverse/copy tokens.",
        "healthy": "Should trend downward within a curriculum stage.",
        "watch": "Sudden spikes after a length jump can be normal; persistent rise is not.",
    },
    "mean_loss": {
        "label": "Epoch running mean loss",
        "group": "core",
        "direction": "lower",
        "unit": "CE",
        "meaning": "Average train loss so far in the current epoch.",
        "healthy": "Smooth downward drift inside a fixed sequence length.",
        "watch": "If mean_loss falls but val loss rises, the model may be overfitting or memorizing noise.",
    },
    "acc": {
        "label": "Token accuracy",
        "group": "core",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Fraction of predicted tokens that match the target (excluding pads where applicable).",
        "healthy": "Should climb with training; len-8 reverse often moves first.",
        "watch": "Flat near chance (~1/vocab) means the decoder is not learning the mapping yet.",
    },
    "seq_acc": {
        "label": "Exact sequence accuracy",
        "group": "core",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Fraction of sequences where every token is correct.",
        "healthy": "Strict metric; rises later than token accuracy.",
        "watch": "Zero for long stretches is common early; worry if token acc rises but seq_acc stays 0 forever.",
    },
    "recall_loss": {
        "label": "Recall auxiliary loss",
        "group": "core",
        "direction": "lower",
        "unit": "loss",
        "meaning": "Memory retrieval / recall objective from cortex auxiliary losses.",
        "healthy": "Usually decreases as memory routing stabilizes.",
        "watch": "Very large values can dominate the main CE if recall_loss_weight is high.",
    },
    "cms_loss": {
        "label": "CMS / CPS auxiliary loss",
        "group": "core",
        "direction": "lower",
        "unit": "loss",
        "meaning": "Consolidated-memory / CPS agreement or regularization term.",
        "healthy": "Small positive values are normal.",
        "watch": "Large jumps may indicate unstable CPS fusion or curriculum-stage changes.",
    },
    "pre_grad_norm": {
        "label": "Pre-clip gradient norm",
        "group": "optimization",
        "direction": "stabilize",
        "unit": "L2",
        "meaning": "Global gradient norm after backward, before normalization/clipping.",
        "healthy": "Finite and not exploding across steps.",
        "watch": "Huge spikes with NaNs/Infs mean instability; constant max clip hits mean aggressive updates.",
    },
    "grad_norm": {
        "label": "Post-control gradient norm",
        "group": "optimization",
        "direction": "stabilize",
        "unit": "L2",
        "meaning": "Gradient norm after target-normalization and safety clipping.",
        "healthy": "Near target_grad_norm when scaling works; otherwise clipped.",
        "watch": "If always equal to pre_grad_norm and far above target, gradient control is ineffective.",
    },
    "lr": {
        "label": "Learning rate",
        "group": "optimization",
        "direction": "schedule",
        "unit": "AdamW lr",
        "meaning": "Current optimizer learning rate after warmup/cosine and phase multipliers.",
        "healthy": "Rises in warmup, then decays.",
        "watch": "Too high early can cause loss spikes; too low late stalls learning.",
    },
    "topology_fitness_ema": {
        "label": "Topology fitness EMA",
        "group": "core",
        "direction": "higher",
        "unit": "EMA score",
        "meaning": "Smoothed topology-manager fitness derived from recent loss dynamics.",
        "healthy": "Gradual increase often accompanies improving loss.",
        "watch": "Falling fitness with rising loss suggests unhealthy memory/topology dynamics.",
    },
    "clip_hit_rate": {
        "label": "Gradient clip hit rate",
        "group": "optimization",
        "direction": "lower",
        "unit": "fraction",
        "meaning": "How often gradient clipping activated this epoch.",
        "healthy": "Occasional hits are fine.",
        "watch": "Near 1.0 continuously means gradients are always oversized; consider lower lr or higher target_grad_norm.",
    },
    "token_entropy": {
        "label": "Prediction entropy",
        "group": "core",
        "direction": "stabilize",
        "unit": "nats",
        "meaning": "Average entropy of the model's token distribution.",
        "healthy": "Starts higher (uncertain) and may fall as predictions sharpen.",
        "watch": "Collapse too early can mean overconfident wrong answers.",
    },
    "seq_len": {
        "label": "Curriculum sequence length",
        "group": "core",
        "direction": "context",
        "unit": "tokens",
        "meaning": "Current training sequence length (for reverse/copy curriculum).",
        "healthy": "Matches the intended curriculum stage.",
        "watch": "Accuracy usually drops when length increases; that is expected.",
    },
}


VALIDATION_METRICS: Dict[str, MetricDef] = {
    "val_acc_curriculum_len": {
        "label": "Val accuracy @ current length",
        "group": "validation",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Held-out token accuracy on the current curriculum length.",
        "healthy": "Primary selection metric in default best-checkpoint mode.",
        "watch": "Train rising while this falls means overfitting or curriculum shock.",
    },
    "val_loss_curriculum_len": {
        "label": "Val loss @ current length",
        "group": "validation",
        "direction": "lower",
        "unit": "CE",
        "meaning": "Held-out loss on the current curriculum length.",
        "healthy": "Should improve with train_mean_loss inside a stage.",
        "watch": "Large train/val gap is a warning.",
    },
    "val_acc_len8": {
        "label": "Val accuracy @ length 8",
        "group": "validation",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Held-out accuracy on fixed length-8 reverse/copy.",
        "healthy": "Should remain stable or improve after longer-length training.",
        "watch": "Collapse after len-16 training indicates forgetting short sequences.",
    },
    "val_acc_len16": {
        "label": "Val accuracy @ length 16",
        "group": "validation",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Held-out accuracy on fixed length-16 reverse/copy.",
        "healthy": "Harder than len-8; rises later.",
        "watch": "Near chance after many steps means the long-sequence path is not learning.",
    },
    "val_seq_acc_curriculum_len": {
        "label": "Val exact-seq accuracy @ current length",
        "group": "validation",
        "direction": "higher",
        "unit": "fraction",
        "meaning": "Exact full-sequence match rate on the current curriculum eval set.",
        "healthy": "Late-rising success metric.",
        "watch": "Token acc without seq acc means partial correct sequences only.",
    },
}


MEMORY_BANK_PREFIXES: Dict[str, str] = {
    "hg_": "Hypergraph (HG) long-term memory bank",
    "cgmn_": "CGMN conformal/geometry memory bank",
    "curved_": "Curved-memory bank",
}


MEMORY_FIELD_MEANINGS: Dict[str, str] = {
    "active_slots": "How many memory slots are currently marked active.",
    "usage_mean": "Average slot-usage counter; rising means slots are being revisited.",
    "usage_max": "Most-used slot counter; very skewed max vs mean can mean slot collapse.",
    "temp": "Softmax temperature for retrieval; lower = sharper attention over slots.",
    "topk_base": "Base top-k slots retrieved for this bank.",
    "transformer_layers": "Configured transformer depth for this bank.",
    "topology_fitness_ema": "Bank-local topology fitness EMA.",
    "topology_mode": "Active topology mode id for this bank.",
    "conformal_b": "Conformal geometry scale parameter for slot geometry.",
    "lb_top1_avg": "Average lightbulb/top-1 intensity for the bank.",
    "importance_mean": "Mean slot importance weights.",
    "holo_rms_mean": "Mean RMS of holographic codes (QH path).",
    "holo_rms_max": "Max holographic RMS; saturating near a hard max may mean energy saturation.",
    "holo_rms_std": "Spread of holographic RMS values.",
    "episodic_enabled": "Whether episodic HG path is enabled (1/0).",
    "geom_w_euclidean": "Geometry router weight for Euclidean head.",
    "geom_w_hyperbolic": "Geometry router weight for hyperbolic head.",
    "geom_w_spherical": "Geometry router weight for spherical head.",
    "geom_w_torus": "Geometry router weight for toroidal head.",
    "geom_w_fractal": "Geometry router weight for fractal head.",
    "geom_w_cp": "Geometry router weight for complex-projective head.",
}


DIAG_FIELD_MEANINGS: Dict[str, str] = {
    "diag_ema_bridge_gate_process": "EMA of WM↔LTM bridge gate during process.",
    "diag_ema_bridge_gate_retrieve": "EMA of WM↔LTM bridge gate during retrieve.",
    "diag_ema_bridge_attn_wm_process": "EMA attention mass from bridge toward WM (process).",
    "diag_ema_bridge_attn_ltm_process": "EMA attention mass from bridge toward LTM (process).",
    "diag_ema_bridge_attn_wm_retrieve": "EMA attention mass from bridge toward WM (retrieve).",
    "diag_ema_bridge_attn_ltm_retrieve": "EMA attention mass from bridge toward LTM (retrieve).",
    "diag_ema_cps_agree_loss": "EMA of CPS head-agreement loss.",
    "diag_ema_fire_rate": "EMA fraction of lightbulb/fire events.",
    "diag_ema_importance_mean": "EMA of memory importance scores.",
    "diag_ema_recall_boost": "EMA of explosive-recall boost factor.",
    "diag_ema_recall_loss": "EMA of recall auxiliary loss.",
    "diag_ema_sensory_norm": "EMA of sensory/context embedding norm.",
    "diag_ema_topology_fitness_ema": "EMA of topology fitness from diagnostics.",
    "diag_ema_write_gate_prob": "EMA probability of write-gate opening.",
    "diag_enabled": "Diagnostics subsystem enabled (1/0).",
    "diag_events_buffered": "How many diagnostic events are buffered.",
}


ROUTER_FIELD_MEANINGS: Dict[str, str] = {
    "ltm_router_hg": "Softmax weight choosing the HG bank.",
    "ltm_router_cgmn": "Softmax weight choosing the CGMN bank.",
    "ltm_router_curved": "Softmax weight choosing the curved bank.",
    "ltm_router_spcp": "Softmax weight choosing SPCP/spatial-like path.",
    "ltm_router_cons_novelty": "Router novelty/consolidation feature.",
    "ltm_router_lightbulb_intensity": "Router lightbulb intensity feature.",
    "ltm_inter_gate": "Inter-bank fusion gate.",
    "ltm_inter_global_mean": "Mean global inter-bank attention.",
    "ltm_inter_hg_to_cg_mean": "Mean attention from HG toward CGMN.",
    "ltm_inter_cg_to_spatial_mean": "Mean attention from CGMN toward spatial.",
    "ltm_inter_spatial_to_hg_mean": "Mean attention from spatial toward HG.",
    "ltm_spatial_enabled": "Spatial LTM enabled flag.",
    "ima_enabled": "Inter-manifold attention is mounted (1/0).",
    "ima_gate": "Residual mix used to inject inter-manifold communications.",
    "ima_entropy": "Entropy of inter-manifold attention; high means broad mixing, low means a few dominant edges.",
    "ima_token_count": "Number of manifold/system tokens in the communication graph.",
    "ima_global_mean": "Mean inter-manifold attention weight.",
    "ima_top_edge_score": "Strongest off-diagonal communication edge this step.",
    "ima_finite": "Whether the inter-manifold mixer produced a finite tensor.",
}


MISC_FIELD_MEANINGS: Dict[str, str] = {
    "adaptive_lr_mult": "Live adaptive LR multiplier if adaptive_lr is enabled.",
    "adaptive_bad_streak": "Consecutive bad-step counter for adaptive LR.",
    "phase_id": "Curriculum phase id (1 early / 2 later).",
    "phase_lr_mult": "Manual phase learning-rate multiplier.",
    "lr_base": "Scheduler base LR before some adaptive adjustments.",
    "grad_scale": "AMP GradScaler scale (FP16); usually 1 for BF16/disabled.",
    "grad_scale_mean": "Running mean of AMP scale.",
    "clip_hit": "Whether this step hit the clip threshold.",
    "ema_loss": "Exponential moving average of loss if tracked.",
    "topology_frozen": "Topology manager frozen flag.",
    "shared_slot_episodic_writes_step": "Episodic shared-slot writes this step.",
    "shared_slot_consolidation_writes_step": "Consolidation shared-slot writes this step.",
    "shared_slot_episodic_writes_total": "Cumulative episodic shared-slot writes.",
    "shared_slot_consolidation_writes_total": "Cumulative consolidation shared-slot writes.",
    "shared_mem_enabled": "Shared-memory subsystem enabled.",
    "qdt_wrapper_enabled": "QDT working-memory wrapper active.",
    "wm_wrapper": "Name of the mounted working-memory wrapper class.",
    "cms_depth_stack_enabled": "CMS depth stack enabled.",
    "hgm_enabled": "Hypergraph manifold bridge enabled.",
    "hgm_assignments": "Number of HGM assignments recorded.",
    "spatial_ltm_extension_enabled": "Spatial LTM extension enabled.",
    "energy_mode": "Energy-saving mode flag.",
    "forgetting_threshold": "Configured forgetting threshold.",
    "threshold": "Live adaptive/topology threshold value.",
    "total_fires": "Cumulative fire/lightbulb events.",
    "total_samples": "Cumulative training samples seen.",
    "trigger_rate_ema": "EMA of fire/trigger rate.",
    "aux_write_gate_prob": "Auxiliary write-gate probability from the forward pass.",
    "aux_recall_loss": "Alias of recall aux loss from model aux dict.",
    "aux_cms_loss": "Alias of CMS aux loss from model aux dict.",
}


def describe_metric(key: str) -> MetricDef:
    if key in CORE_TERMINAL_METRICS:
        return CORE_TERMINAL_METRICS[key]
    if key in VALIDATION_METRICS:
        return VALIDATION_METRICS[key]
    if key in DIAG_FIELD_MEANINGS:
        return {
            "label": key,
            "group": "diagnostics",
            "direction": "context",
            "unit": "",
            "meaning": DIAG_FIELD_MEANINGS[key],
            "healthy": "Interpret relative trends, not absolute targets.",
            "watch": "Sudden regime changes after curriculum length jumps are common.",
        }
    if key in ROUTER_FIELD_MEANINGS:
        return {
            "label": key,
            "group": "ltm_router",
            "direction": "context",
            "unit": "weight/feature",
            "meaning": ROUTER_FIELD_MEANINGS[key],
            "healthy": "Bank weights should not permanently collapse to a single bank unless intentional.",
            "watch": "One bank >0.8 for long stretches can starve other memories.",
        }
    if key in MISC_FIELD_MEANINGS:
        return {
            "label": key,
            "group": "misc",
            "direction": "context",
            "unit": "",
            "meaning": MISC_FIELD_MEANINGS[key],
            "healthy": "Depends on feature flags.",
            "watch": "Unexpected non-zero writes while writes are disabled is a bug signal.",
        }
    for prefix, bank in MEMORY_BANK_PREFIXES.items():
        if key.startswith(prefix):
            field = key[len(prefix) :]
            meaning = MEMORY_FIELD_MEANINGS.get(field, f"{bank} field `{field}`.")
            return {
                "label": key,
                "group": "memory_bank",
                "direction": "context",
                "unit": "",
                "meaning": f"{bank}: {meaning}",
                "healthy": "Geometry weights should remain a soft distribution; usage should grow with training.",
                "watch": "Usage collapse or one geometry weight dominating permanently can reduce capacity use.",
            }
    return {
        "label": key,
        "group": "other",
        "direction": "context",
        "unit": "",
        "meaning": "Trainer/cortex diagnostic field without a dedicated glossary entry.",
        "healthy": "Use trends and comparison across steps.",
        "watch": "Ignore unless it changes sharply with loss/accuracy.",
    }


def _series(events: Sequence[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for event in events:
        value = event.get(key)
        if isinstance(value, (int, float)) and value == value:
            out.append(float(value))
    return out


def _trend(values: Sequence[float]) -> str:
    if len(values) < 3:
        return "insufficient_history"
    head = sum(values[: max(1, len(values) // 3)]) / max(1, len(values) // 3)
    tail = sum(values[-max(1, len(values) // 3) :]) / max(1, len(values) // 3)
    delta = tail - head
    scale = max(1e-6, abs(head))
    if abs(delta) / scale < 0.03:
        return "flat"
    return "up" if delta > 0 else "down"


def interpret_training(
    events: Sequence[Dict[str, Any]],
    *,
    target_grad_norm: float = 0.8,
) -> Dict[str, Any]:
    steps = [e for e in events if e.get("kind") == "train_step"]
    epochs = [e for e in events if e.get("kind") == "epoch_end"]
    start = next((e for e in events if e.get("kind") == "run_start"), None)
    latest = steps[-1] if steps else (epochs[-1] if epochs else start)
    advice: List[Dict[str, str]] = []

    loss_s = _series(steps, "loss")
    mean_loss_s = _series(steps, "mean_loss")
    acc_s = _series(steps, "acc")
    seq_acc_s = _series(steps, "seq_acc")
    pre_grad_s = _series(steps, "pre_grad_norm")
    clip_s = _series(steps, "clip_hit_rate")
    recall_s = _series(steps, "recall_loss")
    topo_s = _series(steps, "topology_fitness_ema")
    lr_s = _series(steps, "lr")

    loss_trend = _trend(mean_loss_s or loss_s)
    acc_trend = _trend(acc_s)

    if loss_trend == "down" and acc_trend in {"up", "flat"}:
        advice.append(
            {
                "tone": "good",
                "title": "Learning signal present",
                "detail": "Running loss is trending down. Keep the current stage long enough for sequence accuracy to catch up.",
            }
        )
    if loss_trend == "up":
        advice.append(
            {
                "tone": "warn",
                "title": "Loss rising",
                "detail": "Mean/step loss is trending up. Check learning rate, curriculum length jump, or gradient explosions.",
            }
        )
    if acc_s and acc_s[-1] < 0.05:
        advice.append(
            {
                "tone": "info",
                "title": "Accuracy still near chance",
                "detail": "Token accuracy is very low. For reverse tasks this is normal in the first tens of steps; worry only if it stays flat after a full curriculum stage.",
            }
        )
    if seq_acc_s and max(seq_acc_s) < 1e-6 and acc_s and acc_s[-1] > 0.25:
        advice.append(
            {
                "tone": "info",
                "title": "Partial token learning",
                "detail": "Token accuracy is moving but exact sequence accuracy is still ~0. The model is learning fragments; continue training or shorten sequences temporarily.",
            }
        )
    if clip_s and clip_s[-1] > 0.85:
        advice.append(
            {
                "tone": "warn",
                "title": "Gradient clipping always on",
                "detail": (
                    f"Clip hit rate is {clip_s[-1]:.2f}. Updates are repeatedly oversized versus "
                    f"target_grad_norm={target_grad_norm}. Consider lowering lr or raising the target gradually."
                ),
            }
        )
    if pre_grad_s and max(pre_grad_s[-5:]) > 1000:
        advice.append(
            {
                "tone": "warn",
                "title": "Large pre-clip gradients",
                "detail": "Recent pre_grad_norm values are very large. Watch for ComplexHalf warnings, unstable QH banks, or too-high lr.",
            }
        )
    if recall_s and recall_s[0] > 0 and recall_s[-1] < 0.25 * recall_s[0]:
        advice.append(
            {
                "tone": "good",
                "title": "Recall aux improving",
                "detail": "Recall loss dropped substantially, so memory retrieval is stabilizing even if token accuracy is still modest.",
            }
        )
    if topo_s and _trend(topo_s) == "up":
        advice.append(
            {
                "tone": "good",
                "title": "Topology fitness rising",
                "detail": "Topology fitness EMA is improving, which usually accompanies healthier loss dynamics.",
            }
        )
    if len(epochs) >= 2:
        first = epochs[0]
        last = epochs[-1]
        a0 = float(first.get("val_acc_curriculum_len", 0.0))
        a1 = float(last.get("val_acc_curriculum_len", 0.0))
        if a1 + 0.02 < a0 and int(last.get("seq_len", last.get("global_step", 0)) or 0) >= 0:
            advice.append(
                {
                    "tone": "info",
                    "title": "Curriculum shock likely",
                    "detail": (
                        f"Validation accuracy at the active curriculum length moved from {a0:.3f} to {a1:.3f}. "
                        "If sequence length increased, this drop can be expected; compare val_acc_len8 vs val_acc_len16."
                    ),
                }
            )
        v8 = float(last.get("val_acc_len8", 0.0))
        v16 = float(last.get("val_acc_len16", 0.0))
        if v8 > 0.2 and v16 < 0.12:
            advice.append(
                {
                    "tone": "info",
                    "title": "Short sequences ahead of long ones",
                    "detail": (
                        f"len8 val acc={v8:.3f} while len16={v16:.3f}. Spend more steps at length 16, "
                        "or use a gentler curriculum ramp before expecting exact reverse mastery."
                    ),
                }
            )

    # Router collapse check
    if latest:
        router = {
            k: float(latest[k])
            for k in ("ltm_router_hg", "ltm_router_cgmn", "ltm_router_curved", "ltm_router_spcp")
            if isinstance(latest.get(k), (int, float))
        }
        if router:
            top_name, top_val = max(router.items(), key=lambda item: item[1])
            if top_val > 0.75:
                advice.append(
                    {
                        "tone": "warn",
                        "title": "LTM router near collapse",
                        "detail": f"{top_name}={top_val:.2f}. One bank is dominating retrieval; monitor whether other banks' usage stalls.",
                    }
                )

    if not advice:
        advice.append(
            {
                "tone": "info",
                "title": "Waiting for clearer trends",
                "detail": "Not enough stable trend yet. Let the run accumulate more logged steps.",
            }
        )

    highlights = {
        "steps_logged": len(steps),
        "epochs_logged": len(epochs),
        "latest_loss": loss_s[-1] if loss_s else None,
        "latest_mean_loss": mean_loss_s[-1] if mean_loss_s else None,
        "latest_acc": acc_s[-1] if acc_s else None,
        "latest_seq_acc": seq_acc_s[-1] if seq_acc_s else None,
        "latest_lr": lr_s[-1] if lr_s else None,
        "latest_pre_grad": pre_grad_s[-1] if pre_grad_s else None,
        "loss_trend": loss_trend,
        "acc_trend": acc_trend,
        "best_val_acc_curriculum": max(
            (_series(epochs, "val_acc_curriculum_len") or [0.0])
        ),
        "latest_seq_len": latest.get("seq_len") if latest else None,
    }
    series = {
        "global_step": [int(e.get("global_step", 0)) for e in steps],
        "loss": loss_s,
        "mean_loss": mean_loss_s,
        "acc": acc_s,
        "seq_acc": seq_acc_s,
        "recall_loss": recall_s,
        "pre_grad_norm": pre_grad_s,
        "lr": lr_s,
        "topology_fitness_ema": topo_s,
    }
    return {
        "highlights": highlights,
        "advice": advice,
        "latest": latest,
        "epochs": epochs,
        "series": series,
        "start": start,
    }


def catalog_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for mapping in (CORE_TERMINAL_METRICS, VALIDATION_METRICS):
        for key, meta in mapping.items():
            rows.append(
                {
                    "key": key,
                    "label": str(meta["label"]),
                    "group": str(meta["group"]),
                    "meaning": str(meta["meaning"]),
                    "healthy": str(meta["healthy"]),
                    "watch": str(meta["watch"]),
                }
            )
    for key, meaning in sorted({**DIAG_FIELD_MEANINGS, **ROUTER_FIELD_MEANINGS, **MISC_FIELD_MEANINGS}.items()):
        meta = describe_metric(key)
        rows.append(
            {
                "key": key,
                "label": str(meta["label"]),
                "group": str(meta["group"]),
                "meaning": meaning,
                "healthy": str(meta["healthy"]),
                "watch": str(meta["watch"]),
            }
        )
    for prefix, bank in MEMORY_BANK_PREFIXES.items():
        for field, meaning in MEMORY_FIELD_MEANINGS.items():
            key = f"{prefix}{field}"
            rows.append(
                {
                    "key": key,
                    "label": key,
                    "group": "memory_bank",
                    "meaning": f"{bank}: {meaning}",
                    "healthy": "Interpret as bank health / utilization, not task accuracy.",
                    "watch": "Collapse, saturation, or unused banks with zero usage growth.",
                }
            )
    return rows
