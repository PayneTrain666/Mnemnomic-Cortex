from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class _ModuleActivation:
    shape: Optional[Tuple[int, ...]]
    mean: float
    std: float
    min: float
    max: float
    abs_mean: float
    zero_fraction: float


def _extract_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _extract_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _extract_tensor(item)
            if found is not None:
                return found
    return None


def _tensor_stats(t: Optional[torch.Tensor]) -> _ModuleActivation:
    if t is None:
        return _ModuleActivation(
            shape=None,
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            abs_mean=0.0,
            zero_fraction=1.0,
        )
    raw = t.detach()
    x = raw.abs().float() if torch.is_complex(raw) else raw.float()
    if x.numel() == 0:
        return _ModuleActivation(
            shape=tuple(x.shape),
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            abs_mean=0.0,
            zero_fraction=1.0,
        )
    zeros = float((x.abs() <= 1e-12).float().mean().item())
    return _ModuleActivation(
        shape=tuple(x.shape),
        mean=float(x.mean().item()),
        std=float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        min=float(x.min().item()),
        max=float(x.max().item()),
        abs_mean=float(x.abs().mean().item()),
        zero_fraction=zeros,
    )


def _iter_reportable_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if not name:
            continue
        n_params = sum(int(p.numel()) for p in module.parameters(recurse=False))
        has_children = any(True for _ in module.children())
        if n_params > 0 or not has_children:
            yield name, module


def _forward_with_sample(model: nn.Module, sample_batch: Any) -> Any:
    if isinstance(sample_batch, dict):
        return model(**sample_batch)
    if isinstance(sample_batch, (list, tuple)):
        return model(*sample_batch)
    return model(sample_batch)


def _build_seq_probe(model: nn.Module, batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    emb = getattr(model, "embedding", None)
    if emb is None or not hasattr(emb, "num_embeddings"):
        return None
    vocab_size = int(getattr(emb, "num_embeddings", 0))
    if vocab_size <= 1:
        return None
    return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)


def _build_dense_probe(model: nn.Module, batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    input_dim = int(getattr(model, "input_dim", 0))
    if input_dim <= 0:
        return None
    return torch.randn(batch_size, seq_len, input_dim, device=device)


def _device_for_model(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def run_model_audit(
    model: nn.Module,
    *,
    sample_batch: Any = None,
    auto_probe: bool = True,
    probe_batch_size: int = 2,
    probe_seq_len: int = 6,
    include_gradients: bool = False,
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    dead_threshold: float = 1e-7,
    near_dead_threshold: float = 1e-4,
) -> Dict[str, Any]:
    """
    Run a structural + optional runtime layer audit.

    No training run is required. By default this uses a lightweight synthetic
    probe batch (when possible) to validate hidden-layer activation paths.
    """
    model_device = _device_for_model(model)
    module_rows: List[Dict[str, Any]] = []
    hooks = []
    activations: Dict[str, _ModuleActivation] = {}
    touched: Dict[str, bool] = {}
    static_inventory: List[Tuple[str, nn.Module]] = list(_iter_reportable_modules(model))

    for name, module in static_inventory:
        touched[name] = False

        def _hook(_module, _inputs, output, _name=name):
            touched[_name] = True
            activations[_name] = _tensor_stats(_extract_tensor(output))

        hooks.append(module.register_forward_hook(_hook))

    output = None
    used_auto_probe = False
    runtime_error = None

    try:
        was_training = bool(model.training)
        if sample_batch is None and auto_probe:
            sample_batch = _build_seq_probe(model, probe_batch_size, probe_seq_len, model_device)
            if sample_batch is None:
                sample_batch = _build_dense_probe(model, probe_batch_size, probe_seq_len, model_device)
            used_auto_probe = sample_batch is not None

        model.zero_grad(set_to_none=True)
        if sample_batch is not None:
            if include_gradients:
                model.train()
                output = _forward_with_sample(model, sample_batch)
                raw_loss = loss_fn(output) if loss_fn is not None else _extract_tensor(output)
                if raw_loss is None:
                    runtime_error = "could not derive tensor loss for backward pass"
                else:
                    loss = raw_loss.float().mean()
                    loss.backward()
            else:
                model.eval()
                with torch.no_grad():
                    output = _forward_with_sample(model, sample_batch)
        if was_training:
            model.train()
        else:
            model.eval()
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in hooks:
            handle.remove()

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    dead_modules: List[str] = []
    near_dead_modules: List[str] = []
    skipped_modules: List[str] = []
    grad_missing_modules: List[str] = []
    active_modules = 0

    for name, module in static_inventory:
        n_params = int(sum(p.numel() for p in module.parameters(recurse=False)))
        n_trainable = int(sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad))
        act = activations.get(name)

        if not touched.get(name, False):
            skipped_modules.append(name)
            stats = _ModuleActivation(None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        else:
            active_modules += 1
            stats = act or _ModuleActivation(None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        status = "ok"
        if touched.get(name, False):
            if stats.abs_mean <= float(dead_threshold):
                status = "dead"
                dead_modules.append(name)
            elif stats.abs_mean <= float(near_dead_threshold):
                status = "near_dead"
                near_dead_modules.append(name)
        else:
            status = "skipped"

        grad_available = None
        grad_abs_mean = None
        if include_gradients:
            grads = []
            for p in module.parameters(recurse=False):
                if not p.requires_grad or p.grad is None:
                    continue
                g = p.grad.detach()
                grads.append(g.abs().float() if torch.is_complex(g) else g.float())
            if grads:
                cat = torch.cat([g.reshape(-1) for g in grads], dim=0)
                grad_available = True
                grad_abs_mean = float(cat.abs().mean().item())
            elif n_trainable > 0:
                grad_available = False
                grad_abs_mean = 0.0
                grad_missing_modules.append(name)

        module_rows.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "parameter_count": n_params,
                "trainable_parameter_count": n_trainable,
                "output_shape": list(stats.shape) if stats.shape is not None else None,
                "activation": {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "abs_mean": stats.abs_mean,
                    "zero_fraction": stats.zero_fraction,
                },
                "status": status,
                "gradient": (
                    {
                        "available": grad_available,
                        "abs_mean": grad_abs_mean,
                    }
                    if include_gradients
                    else None
                ),
            }
        )

    total_modules = len(static_inventory)
    utilization_ratio = float(active_modules / total_modules) if total_modules else 0.0
    recommendations: List[str] = []

    if sample_batch is None:
        recommendations.append(
            "Provide a sample batch or enable auto_probe to collect activation and skip-path evidence."
        )
    if skipped_modules:
        recommendations.append(
            f"{len(skipped_modules)} modules were skipped in the probe path; review conditionals or gating."
        )
    if dead_modules:
        recommendations.append(
            f"{len(dead_modules)} modules appear dead (abs mean <= {dead_threshold:g}); inspect init, normalization, or gate saturation."
        )
    if near_dead_modules:
        recommendations.append(
            f"{len(near_dead_modules)} modules are near-dead (abs mean <= {near_dead_threshold:g}); consider stronger signals or regularization."
        )
    if include_gradients and grad_missing_modules:
        recommendations.append(
            f"{len(grad_missing_modules)} trainable modules had no gradients in this probe; verify loss connectivity."
        )
    if runtime_error:
        recommendations.append(f"Probe execution raised an error: {runtime_error}")
    if not recommendations:
        recommendations.append("No critical activation or gradient issues detected in the current audit probe.")

    header = [
        "Model Audit Report",
        "=" * 18,
        f"Model type: {model.__class__.__name__}",
        f"Total parameters: {total_params:,} (trainable: {trainable_params:,})",
        f"Modules tracked: {total_modules}, activated in probe: {active_modules} ({utilization_ratio:.1%})",
        f"Probe mode: {'auto synthetic batch' if used_auto_probe else ('provided sample batch' if sample_batch is not None else 'static only')}",
        f"Gradient audit: {'enabled' if include_gradients else 'disabled'}",
    ]
    layer_lines = []
    for row in module_rows:
        shape = row["output_shape"] if row["output_shape"] is not None else "n/a"
        act = row["activation"]
        grad_text = ""
        if include_gradients:
            grad = row["gradient"]
            grad_text = f", grad={'yes' if grad and grad['available'] else 'no'}"
            if grad is not None and grad["available"]:
                grad_text += f"({grad['abs_mean']:.3e})"
        layer_lines.append(
            f"- {row['name']} [{row['type']}]: params={row['parameter_count']}, "
            f"out={shape}, abs_mean={act['abs_mean']:.3e}, zero_frac={act['zero_fraction']:.2f}, "
            f"status={row['status']}{grad_text}"
        )

    summary_text = "\n".join(
        header
        + ["", "Per-layer visibility:"]
        + layer_lines
        + ["", "Recommendations:"]
        + [f"- {line}" for line in recommendations]
    )

    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "probe": {
            "used_sample_batch": sample_batch is not None,
            "used_auto_probe": used_auto_probe,
            "include_gradients": bool(include_gradients),
            "runtime_error": runtime_error,
        },
        "utilization": {
            "total_modules": total_modules,
            "active_modules": active_modules,
            "utilization_ratio": utilization_ratio,
            "skipped_modules": skipped_modules,
            "dead_modules": dead_modules,
            "near_dead_modules": near_dead_modules,
            "gradient_missing_modules": grad_missing_modules,
        },
        "layers": module_rows,
        "recommendations": recommendations,
        "text_report": summary_text,
    }
