import json
import os
import time
from typing import Any, Dict

import torch


class ParameterAuditLogger:
    """
    Logs parameter inventories and dynamic parameter creation events.
    """

    def __init__(self, log_path: str = "param_audit.jsonl", max_dynamic_events: int = 5000):
        self.log_path = log_path
        self.max_dynamic_events = int(max_dynamic_events)
        self.dynamic_events = 0
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    @staticmethod
    def _tensor_summary(t: torch.Tensor, include_preview: bool = True, preview_n: int = 12):
        raw = t.detach().reshape(-1)
        if torch.is_complex(raw):
            # Log magnitude statistics for complex tensors to avoid lossy casts.
            x = raw.abs().float()
            preview_vals = [
                {"re": float(v.real.item()), "im": float(v.imag.item())}
                for v in raw[: min(preview_n, raw.numel())]
            ]
        else:
            x = raw.float()
            preview_vals = [float(v) for v in x[: min(preview_n, x.numel())].tolist()]
        out = {
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "requires_grad": bool(getattr(t, "requires_grad", False)),
            "mean": float(x.mean().item()) if x.numel() else 0.0,
            "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            "min": float(x.min().item()) if x.numel() else 0.0,
            "max": float(x.max().item()) if x.numel() else 0.0,
        }
        if include_preview:
            out["preview"] = preview_vals
        return out

    def _write(self, rec: Dict[str, Any]):
        rec = {"ts": time.time(), **rec}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def log_model_inventory(self, model, tag: str = "startup", include_preview: bool = True):
        params = []
        total = 0
        trainable = 0
        for name, p in model.named_parameters():
            total += int(p.numel())
            if p.requires_grad:
                trainable += int(p.numel())
            params.append(
                {
                    "name": name,
                    **self._tensor_summary(p, include_preview=include_preview),
                }
            )
        self._write(
            {
                "event": "model_parameter_inventory",
                "tag": tag,
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": total - trainable,
                "parameters": params,
            }
        )
        return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}

    def log_dynamic_creation(self, payload: Dict[str, Any]):
        if self.dynamic_events >= self.max_dynamic_events:
            return
        self.dynamic_events += 1
        self._write({"event": "dynamic_parameter_creation", "index": self.dynamic_events, **payload})

    def attach_to_cortex(self, cortex):
        # CPS dynamic creations
        if hasattr(cortex, "cps") and hasattr(cortex.cps, "register_creation_hook"):
            cortex.cps.register_creation_hook(self.log_dynamic_creation)
        # Advanced CMS/CPS dynamic creations (if enabled)
        if hasattr(cortex, "advanced_cms") and cortex.advanced_cms is not None and hasattr(cortex.advanced_cms, "register_creation_hook"):
            cortex.advanced_cms.register_creation_hook(self.log_dynamic_creation)
        if hasattr(cortex, "multi_cps") and cortex.multi_cps is not None:
            for cps in cortex.multi_cps.cps.values():
                if hasattr(cps, "register_creation_hook"):
                    cps.register_creation_hook(self.log_dynamic_creation)

