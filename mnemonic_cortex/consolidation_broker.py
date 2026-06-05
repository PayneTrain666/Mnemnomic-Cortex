from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consolidated_lexicon import ConsolidatedLexicon


@dataclass
class StoreConfig:
    senses: int = 3
    conformal_b: float = 0.02
    w_h: float = 0.6
    w_p: float = 0.25
    w_e: float = 0.15


class ConsolidationBroker(nn.Module):
    """
    Phase-A broker: route token-level CMS fusion across purpose-specific stores.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        store_configs: Optional[Dict[str, StoreConfig]] = None,
    ):
        super().__init__()
        defaults = {
            "CRS": StoreConfig(senses=3, conformal_b=0.025, w_h=0.65, w_p=0.2, w_e=0.15),
            "SKS": StoreConfig(senses=4, conformal_b=0.02, w_h=0.45, w_p=0.35, w_e=0.2),
            "CAS": StoreConfig(senses=4, conformal_b=0.03, w_h=0.3, w_p=0.5, w_e=0.2),
            "PSS": StoreConfig(senses=3, conformal_b=0.02, w_h=0.4, w_p=0.2, w_e=0.4),
        }
        if store_configs:
            defaults.update(store_configs)
        self.store_configs = defaults
        self.stores = nn.ModuleDict(
            {
                name: ConsolidatedLexicon(
                    vocab_size=vocab_size,
                    model_dim=model_dim,
                    senses=cfg.senses,
                    conformal_b=cfg.conformal_b,
                    w_h=cfg.w_h,
                    w_p=cfg.w_p,
                    w_e=cfg.w_e,
                    context_dim=model_dim,
                )
                for name, cfg in defaults.items()
            }
        )
        self.register_buffer("_store_hits", torch.zeros(len(self.stores), dtype=torch.long))
        self._store_names: List[str] = list(self.stores.keys())
        # Logical aliases for additional purpose-specific stores without duplicating
        # full parameter banks on memory-constrained hardware.
        self.store_alias = {
            "CGS": "SKS",   # Causal Graph Store -> science/fisher-heavy
            "MSS": "CRS",   # Math/Symbolic Store -> reasoning tree-heavy
            "TEPS": "PSS",  # Temporal planning -> procedural dynamics
            "CSS": "CRS",   # Commonsense scripts -> reasoning/script structures
            "SNS": "SKS",   # Safety/policy -> evidence-grounded checks
            "MLS": "PSS",   # Meta-learning strategy -> procedural priors
        }
        self.intent_map = {
            "reason": ["CRS"],
            "reasoning": ["CRS"],
            "science": ["SKS"],
            "scientific": ["SKS"],
            "creative": ["CAS"],
            "creativity": ["CAS"],
            "skill": ["PSS"],
            "skills": ["PSS"],
            "procedural": ["PSS"],
            "causal": ["CGS"],
            "math": ["MSS"],
            "symbolic": ["MSS"],
            "planning": ["TEPS"],
            "commonsense": ["CSS"],
            "safety": ["SNS"],
            "meta": ["MLS"],
            "auto": ["CRS", "SKS"],
        }
        self._unified: Dict[str, Dict[str, Any]] = {}

    def _choose_stores(self, intent: str) -> List[str]:
        key = str(intent or "auto").lower()
        if key in self.intent_map:
            return self.intent_map[key]
        return self.intent_map["auto"]

    def _resolve_store_name(self, logical_name: str) -> str:
        return self.store_alias.get(logical_name, logical_name)

    def _store_score(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Per-token confidence proxy: smaller distances and lower warp deviation.
        d_h = aux["dist_h"].mean(dim=-1)
        d_p = aux["dist_p"].mean(dim=-1)
        d_e = aux["dist_e"].mean(dim=-1)
        warp_dev = (aux["warp"] - 1.0).abs()
        return -(d_h + d_p + d_e + 0.25 * warp_dev)

    def route_fuse(
        self,
        token_ids: torch.Tensor,
        base_embed: torch.Tensor,
        context_feat: Optional[torch.Tensor] = None,
        intent: str = "auto",
    ) -> Tuple[torch.Tensor, Dict]:
        chosen = self._choose_stores(intent)
        fused_list = []
        score_list = []
        aux_by_store = {}
        seen_physical = set()
        for name in chosen:
            physical = self._resolve_store_name(name)
            if physical in seen_physical:
                continue
            seen_physical.add(physical)
            store = self.stores[physical]
            fused, _, aux = store(token_ids, base_embed, context_feat)
            fused_list.append(fused)
            score_list.append(self._store_score(aux))
            aux_by_store[name] = aux
            idx = self._store_names.index(physical)
            self._store_hits[idx] += 1

        if len(fused_list) == 1:
            return fused_list[0], {"selected_store": chosen[0], "weights": None, "stores": aux_by_store}

        scores = torch.stack(score_list, dim=-1)  # [N, S]
        weights = torch.softmax(scores, dim=-1)
        fused = torch.zeros_like(fused_list[0])
        for i, f in enumerate(fused_list):
            fused = fused + weights[:, i].unsqueeze(-1) * f

        best_idx = weights.mean(dim=0).argmax().item()
        selected = chosen[best_idx]
        return fused, {
            "selected_store": selected,
            "weights": weights,
            "stores": aux_by_store,
        }

    @staticmethod
    def _unit_complex_real(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        q2 = z.size(-1)
        if q2 % 2 != 0:
            raise ValueError(f"Expected even last dim for packed complex, got {q2}")
        zc = torch.view_as_complex(z.view(*z.shape[:-1], q2 // 2, 2).contiguous())
        n = torch.linalg.norm(zc, dim=-1, keepdim=True).clamp_min(eps)
        zc = zc / n
        return torch.view_as_real(zc).reshape_as(z)

    @torch.no_grad()
    def _headspace_from_items(self, store: ConsolidatedLexicon, items: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Multi-memory -> unified per-store representation in the store's head space.
        cue_h = torch.tanh(store.cue_h(items)).clamp(-0.95, 0.95)
        cue_p = self._unit_complex_real(store.cue_p(items))
        cue_e = store.cue_e(items)
        return {
            "h": cue_h.mean(dim=0),
            "p": self._unit_complex_real(cue_p.mean(dim=0, keepdim=True)).squeeze(0),
            "e": cue_e.mean(dim=0),
            "n": int(items.size(0)),
        }

    @torch.no_grad()
    def unify_and_route_write(self, items: torch.Tensor, metas: List[dict], domain: str) -> None:
        if items.numel() == 0:
            return
        key = str(domain or "reasoning").lower()
        domain_map = {
            "reasoning": "CRS", "logic": "CRS", "math": "CRS", "proof": "CRS",
            "science": "SKS", "stem": "SKS", "facts": "SKS",
            "creativity": "CAS", "analogy": "CAS", "style": "CAS",
            "skills": "PSS", "tools": "PSS", "procedure": "PSS",
            "causal": "CGS", "symbolic": "MSS", "planning": "TEPS",
            "commonsense": "CSS", "safety": "SNS", "meta": "MLS",
        }
        store_name = self._resolve_store_name(domain_map.get(key, "CRS"))
        store = self.stores[store_name]

        items = items.to(next(store.parameters()).device)
        heads = self._headspace_from_items(store, items)
        rec = self._unified.get(store_name)
        if rec is None:
            merged_meta = {"domain": domain, "sources": [], "tags": [], "time": None}
            for m in metas:
                if not m:
                    continue
                merged_meta["sources"].extend(m.get("sources", []))
                merged_meta["tags"].extend(m.get("tags", []))
                t = m.get("time")
                if t is not None:
                    merged_meta["time"] = t if merged_meta["time"] is None else min(merged_meta["time"], t)
            merged_meta["sources"] = list(dict.fromkeys(merged_meta["sources"]))
            merged_meta["tags"] = list(dict.fromkeys(merged_meta["tags"]))
            self._unified[store_name] = {"heads": heads, "meta": merged_meta, "updates": 1}
            return

        # Precision-style EMA by sample counts.
        prev_n = float(max(1, rec["heads"].get("n", 1)))
        cur_n = float(max(1, heads.get("n", 1)))
        alpha = cur_n / (prev_n + cur_n)
        rec["heads"]["h"] = (1.0 - alpha) * rec["heads"]["h"] + alpha * heads["h"]
        rec["heads"]["p"] = self._unit_complex_real(
            ((1.0 - alpha) * rec["heads"]["p"] + alpha * heads["p"]).unsqueeze(0)
        ).squeeze(0)
        rec["heads"]["e"] = (1.0 - alpha) * rec["heads"]["e"] + alpha * heads["e"]
        rec["heads"]["n"] = int(prev_n + cur_n)
        rec["updates"] = int(rec.get("updates", 0) + 1)

    @torch.no_grad()
    def route_read(self, query: torch.Tensor, intent: str = "auto", k: int = 8) -> Dict[str, Any]:
        # Query-level diagnostics for AHG and recall gating.
        _ = k
        chosen = self._choose_stores(intent)
        best = None
        debug: Dict[str, Any] = {}
        for name in chosen:
            physical = self._resolve_store_name(name)
            store = self.stores[physical]
            q = query.to(next(store.parameters()).device)
            qh = torch.tanh(store.cue_h(q)).clamp(-0.95, 0.95)
            qp = self._unit_complex_real(store.cue_p(q))
            qe = store.cue_e(q)

            rec = self._unified.get(name)
            if rec is None:
                proto = torch.full((q.size(0),), 1e3, device=q.device)
                phase_agree = torch.zeros(q.size(0), device=q.device)
                fisher_unc = torch.full((q.size(0),), 1e3, device=q.device)
            else:
                uh, up, ue = rec["heads"]["h"], rec["heads"]["p"], rec["heads"]["e"]
                dh = torch.norm(qh - uh.unsqueeze(0), dim=-1)
                dp = torch.norm(qp - up.unsqueeze(0), dim=-1)
                de = torch.norm(qe - ue.unsqueeze(0), dim=-1)
                cfg = self.store_configs[name]
                proto = cfg.w_h * dh + cfg.w_p * dp + cfg.w_e * de
                phase_agree = 1.0 / (1.0 + dp)
                fisher_unc = de

            sc = (-proto).mean().item()
            payload = {
                "score": sc,
                "proto_distance": float(proto.mean().item()),
                "phase_agreement": float(phase_agree.mean().item()),
                "fisher_uncertainty": float(fisher_unc.mean().item()),
            }
            debug[name] = dict(payload, physical_store=physical)
            if best is None or sc > best["score"]:
                best = {"name": name, "physical_store": physical, **payload}

        if best is None:
            return {
                "store": "none",
                "values": None,
                "indices": None,
                "signals": {
                    "proto_distance": 1e3,
                    "phase_agreement": 0.0,
                    "fisher_uncertainty": 1e3,
                    "has_fisher": False,
                },
                "debug": debug,
            }
        return {
            "store": best["name"],
            "physical_store": best.get("physical_store", best["name"]),
            "values": None,
            "indices": None,
            "signals": {
                "proto_distance": best["proto_distance"],
                "phase_agreement": best["phase_agreement"],
                "fisher_uncertainty": best["fisher_uncertainty"],
                "has_fisher": True,
            },
            "debug": debug,
        }

    @torch.no_grad()
    def cross_store_diagnostics(self, query: torch.Tensor, k: int = 8) -> Dict[str, float]:
        _ = k
        names = list(self.stores.keys())
        if len(names) < 2:
            return {"agreement": 0.0}
        protos = []
        for name in names:
            out = self.route_read(query, intent=name, k=k)
            protos.append(float(out["signals"]["proto_distance"]))
        t = torch.tensor(protos, dtype=torch.float32)
        # Low variance in selected proto distances => stronger cross-store agreement.
        agreement = float((1.0 / (1.0 + t.std(unbiased=False))).item())
        return {"agreement": agreement}

    @torch.no_grad()
    def renorm_constraints_(self):
        for store in self.stores.values():
            store.renorm_constraints_()

    @torch.no_grad()
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self._store_names):
            cfg = self.store_configs[name]
            rec = self._unified.get(name)
            out[name] = {
                "hits": float(self._store_hits[i].item()),
                "senses": float(cfg.senses),
                "conformal_b": float(cfg.conformal_b),
                "w_h": float(cfg.w_h),
                "w_p": float(cfg.w_p),
                "w_e": float(cfg.w_e),
                "unified_n": float(rec["heads"]["n"]) if rec is not None else 0.0,
                "unified_updates": float(rec.get("updates", 0)) if rec is not None else 0.0,
            }
        # expose alias mapping for interpretability
        for logical, physical in self.store_alias.items():
            out[f"alias_{logical}"] = {"maps_to": float(self._store_names.index(physical))}
        return out

