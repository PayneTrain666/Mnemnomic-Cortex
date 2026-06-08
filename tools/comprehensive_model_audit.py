"""Comprehensive neural-layer and parameter-space audit for Mnemonic Cortex models."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.model_audit import run_model_audit
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


@dataclass
class HiddenModuleRecord:
    path: str
    type_name: str
    parameter_count: int
    trainable_parameter_count: int


@dataclass
class HiddenTensorRecord:
    path: str
    shape: List[int]
    dtype: str
    requires_grad: bool
    numel: int
    nbytes: int
    device: str


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _format_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, int(nbytes)))
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _collect_registered_ids(model: nn.Module) -> Tuple[Set[int], Set[int], Set[int]]:
    param_ids = {id(p) for p in model.parameters()}
    buffer_ids = {id(b) for b in model.buffers()}
    module_ids = {id(m) for _, m in model.named_modules()}
    return param_ids, buffer_ids, module_ids


def _iter_public_attrs(obj: Any) -> Iterable[Tuple[str, Any]]:
    if not hasattr(obj, "__dict__"):
        return []
    for key, value in vars(obj).items():
        if key.startswith("_"):
            continue
        yield key, value


def _scan_hidden_elements(model: nn.Module) -> Dict[str, Any]:
    param_ids, buffer_ids, module_ids = _collect_registered_ids(model)
    known_tensor_ids = set(param_ids | buffer_ids)
    seen_obj_ids: Set[int] = set()

    hidden_modules: List[HiddenModuleRecord] = []
    hidden_tensors: List[HiddenTensorRecord] = []

    def walk(value: Any, path: str, depth: int) -> None:
        if depth > 12:
            return
        obj_id = id(value)
        if obj_id in seen_obj_ids:
            return
        seen_obj_ids.add(obj_id)

        if isinstance(value, nn.Module):
            if obj_id not in module_ids:
                hidden_modules.append(
                    HiddenModuleRecord(
                        path=path,
                        type_name=value.__class__.__name__,
                        parameter_count=int(sum(p.numel() for p in value.parameters())),
                        trainable_parameter_count=int(
                            sum(p.numel() for p in value.parameters() if p.requires_grad)
                        ),
                    )
                )
            # Traverse registered module hierarchy explicitly: nn.Module stores child
            # modules in _modules (not in __dict__ public attrs).
            for child_name, child in value.named_children():
                walk(child, f"{path}.{child_name}", depth + 1)
            for key, child in _iter_public_attrs(value):
                walk(child, f"{path}.{key}", depth + 1)
            return

        if isinstance(value, torch.Tensor):
            if obj_id not in known_tensor_ids:
                hidden_tensors.append(
                    HiddenTensorRecord(
                        path=path,
                        shape=list(value.shape),
                        dtype=_dtype_name(value.dtype),
                        requires_grad=bool(value.requires_grad),
                        numel=int(value.numel()),
                        nbytes=int(value.numel() * value.element_size()),
                        device=str(value.device),
                    )
                )
            return

        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}[{repr(key)}]", depth + 1)
            return

        if isinstance(value, (list, tuple, set)):
            for idx, child in enumerate(value):
                walk(child, f"{path}[{idx}]", depth + 1)
            return

        if hasattr(value, "__dict__"):
            for key, child in _iter_public_attrs(value):
                walk(child, f"{path}.{key}", depth + 1)

    walk(model, "model", 0)

    hidden_modules.sort(key=lambda r: (-r.parameter_count, r.path))
    hidden_tensors.sort(key=lambda r: (-r.numel, r.path))
    hidden_module_param_total = int(sum(r.parameter_count for r in hidden_modules))
    hidden_module_trainable_total = int(sum(r.trainable_parameter_count for r in hidden_modules))
    hidden_tensor_numel_total = int(sum(r.numel for r in hidden_tensors))
    hidden_tensor_nbytes_total = int(sum(r.nbytes for r in hidden_tensors))

    return {
        "hidden_modules": [r.__dict__ for r in hidden_modules],
        "hidden_tensors": [r.__dict__ for r in hidden_tensors],
        "totals": {
            "hidden_module_count": len(hidden_modules),
            "hidden_module_parameter_count": hidden_module_param_total,
            "hidden_module_trainable_parameter_count": hidden_module_trainable_total,
            "hidden_tensor_count": len(hidden_tensors),
            "hidden_tensor_numel": hidden_tensor_numel_total,
            "hidden_tensor_nbytes": hidden_tensor_nbytes_total,
        },
    }


def _parameter_space_breakdown(model: nn.Module) -> Dict[str, Any]:
    by_dtype = defaultdict(lambda: {"parameters": 0, "nbytes": 0, "trainable_parameters": 0})
    by_top = defaultdict(lambda: {"parameters": 0, "trainable_parameters": 0, "nbytes": 0})
    total_nbytes = 0

    for name, p in model.named_parameters():
        dtype = _dtype_name(p.dtype)
        n = int(p.numel())
        b = int(p.numel() * p.element_size())
        top = name.split(".")[0] if "." in name else name
        by_dtype[dtype]["parameters"] += n
        by_dtype[dtype]["nbytes"] += b
        by_top[top]["parameters"] += n
        by_top[top]["nbytes"] += b
        if p.requires_grad:
            by_dtype[dtype]["trainable_parameters"] += n
            by_top[top]["trainable_parameters"] += n
        total_nbytes += b

    buffer_total = int(sum(int(b.numel()) for b in model.buffers()))
    buffer_nbytes = int(sum(int(b.numel() * b.element_size()) for b in model.buffers()))

    return {
        "parameter_nbytes": total_nbytes,
        "parameter_nbytes_human": _format_bytes(total_nbytes),
        "buffer_numel": buffer_total,
        "buffer_nbytes": buffer_nbytes,
        "buffer_nbytes_human": _format_bytes(buffer_nbytes),
        "by_dtype": dict(sorted(by_dtype.items(), key=lambda kv: -kv[1]["parameters"])),
        "by_top_module": dict(sorted(by_top.items(), key=lambda kv: -kv[1]["parameters"])),
    }


def _build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "enhanced_mnemonic_cortex":
        return EnhancedMnemonicCortex(input_dim=args.input_dim, output_dim=args.output_dim)
    if args.model == "enhanced_triple_hybrid":
        return EnhancedTripleHybridMemory(input_dim=args.input_dim, output_dim=args.output_dim)
    raise ValueError(f"unknown model '{args.model}'")


def _sample_batch_for_model(model_name: str, batch_size: int, seq_len: int, dim: int) -> Any:
    x = torch.randn(batch_size, seq_len, dim)
    if model_name == "enhanced_mnemonic_cortex":
        context = torch.randn(batch_size, dim)
        return (x, context)
    return (x,)


def _make_report_markdown(audit: Dict[str, Any]) -> str:
    meta = audit["metadata"]
    core = audit["audit"]
    util = core["utilization"]
    param = audit["parameter_space"]
    hidden = audit["hidden_elements"]
    layers = core["layers"]

    visible_layers = [r for r in layers if r["status"] != "skipped"]
    skipped_layers = [r for r in layers if r["status"] == "skipped"]
    dead_layers = [r for r in layers if r["status"] == "dead"]
    near_dead_layers = [r for r in layers if r["status"] == "near_dead"]
    largest_layers = sorted(layers, key=lambda r: int(r["parameter_count"]), reverse=True)[:25]

    lines: List[str] = []
    lines.append("# Comprehensive Neural Model Audit")
    lines.append("")
    lines.append("## Audit Scope")
    lines.append(f"- Generated at (UTC): `{meta['generated_utc']}`")
    lines.append(f"- Model target: `{meta['model_name']}`")
    lines.append(f"- Runtime model type: `{core['model_type']}`")
    lines.append(
        f"- Probe batch configuration: batch={meta['probe']['batch_size']}, seq={meta['probe']['seq_len']}, dim={meta['probe']['input_dim']}"
    )
    lines.append(f"- Gradient connectivity audit: `{meta['probe']['include_gradients']}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Total parameter space: `{_format_int(core['total_parameters'])}` parameters")
    lines.append(f"- Trainable parameters: `{_format_int(core['trainable_parameters'])}`")
    lines.append(
        f"- Frozen/non-trainable parameters: `{_format_int(core['total_parameters'] - core['trainable_parameters'])}`"
    )
    lines.append(
        f"- Registered module utilization in probe: `{util['active_modules']}/{util['total_modules']}` ({util['utilization_ratio']:.2%})"
    )
    lines.append(f"- Visible (activated) layers: `{len(visible_layers)}`")
    lines.append(f"- Hidden/skipped registered layers: `{len(skipped_layers)}`")
    lines.append(f"- Dead layers (`abs_mean <= threshold`): `{len(dead_layers)}`")
    lines.append(f"- Near-dead layers: `{len(near_dead_layers)}`")
    lines.append(f"- Hidden/unregistered module objects: `{hidden['totals']['hidden_module_count']}`")
    lines.append(f"- Hidden/unregistered tensors: `{hidden['totals']['hidden_tensor_count']}`")
    lines.append("")
    lines.append("## Parameter Space Quantification")
    lines.append(f"- Parameter tensor footprint: `{param['parameter_nbytes_human']}`")
    lines.append(f"- Buffer tensor footprint: `{param['buffer_nbytes_human']}`")
    lines.append(f"- Buffer element count: `{_format_int(param['buffer_numel'])}`")
    lines.append("")
    lines.append("### Parameter Space by Top-Level Module")
    for name, stats in param["by_top_module"].items():
        lines.append(
            f"- `{name}`: params={_format_int(stats['parameters'])}, trainable={_format_int(stats['trainable_parameters'])}, bytes={_format_bytes(stats['nbytes'])}"
        )
    lines.append("")
    lines.append("### Parameter Space by DType")
    for dtype, stats in param["by_dtype"].items():
        lines.append(
            f"- `{dtype}`: params={_format_int(stats['parameters'])}, trainable={_format_int(stats['trainable_parameters'])}, bytes={_format_bytes(stats['nbytes'])}"
        )
    lines.append("")
    lines.append("## Hidden Elements (Unregistered / Non-Standard State)")
    lines.append(
        f"- Hidden module parameter total: `{_format_int(hidden['totals']['hidden_module_parameter_count'])}`"
    )
    lines.append(
        f"- Hidden module trainable parameter total: `{_format_int(hidden['totals']['hidden_module_trainable_parameter_count'])}`"
    )
    lines.append(
        f"- Hidden tensor total elements: `{_format_int(hidden['totals']['hidden_tensor_numel'])}`"
    )
    lines.append(
        f"- Hidden tensor memory: `{_format_bytes(hidden['totals']['hidden_tensor_nbytes'])}`"
    )
    lines.append("")
    lines.append("### Hidden Unregistered Modules")
    if hidden["hidden_modules"]:
        for mod in hidden["hidden_modules"]:
            lines.append(
                f"- `{mod['path']}` (`{mod['type_name']}`): params={_format_int(mod['parameter_count'])}, trainable={_format_int(mod['trainable_parameter_count'])}"
            )
    else:
        lines.append("- None detected.")
    lines.append("")
    lines.append("### Hidden Unregistered Tensors")
    if hidden["hidden_tensors"]:
        for tensor in hidden["hidden_tensors"]:
            lines.append(
                f"- `{tensor['path']}`: shape={tensor['shape']}, dtype={tensor['dtype']}, requires_grad={tensor['requires_grad']}, numel={_format_int(tensor['numel'])}, bytes={_format_bytes(tensor['nbytes'])}, device={tensor['device']}"
            )
    else:
        lines.append("- None detected.")
    lines.append("")
    lines.append("## Largest Registered Layers by Parameter Count")
    for row in largest_layers:
        lines.append(
            f"- `{row['name']}` (`{row['type']}`): params={_format_int(row['parameter_count'])}, trainable={_format_int(row['trainable_parameter_count'])}, status={row['status']}, output_shape={row['output_shape']}"
        )
    lines.append("")
    lines.append("## Registered Layer Visibility (Complete)")
    for row in layers:
        act = row["activation"]
        lines.append(
            f"- `{row['name']}` ({row['type']}): params={_format_int(row['parameter_count'])}, out={row['output_shape']}, status={row['status']}, abs_mean={act['abs_mean']:.6e}, zero_fraction={act['zero_fraction']:.6f}"
        )
    lines.append("")
    lines.append("## Recommendations (Auto-Generated)")
    for rec in core["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- This report separates *registered layers* (from `named_modules`) from hidden/unregistered objects discovered via object-graph traversal."
    )
    lines.append(
        "- Parameter space here refers to the full cardinality of model parameter tensors (`numel`) and related memory footprint."
    )
    lines.append(
        "- Hidden/skipped registered layers are not necessarily bugs; some are path-dependent and may require alternate runtime branches to activate."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a comprehensive layer + parameter audit report.")
    parser.add_argument(
        "--model",
        default="enhanced_mnemonic_cortex",
        choices=["enhanced_mnemonic_cortex", "enhanced_triple_hybrid"],
    )
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--include-gradients", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default="reports/model_comprehensive_audit.json")
    parser.add_argument("--output-md", default="reports/model_comprehensive_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    model = _build_model(args)
    sample_batch = _sample_batch_for_model(args.model, args.batch_size, args.seq_len, args.input_dim)
    loss_fn = None
    if args.include_gradients:
        loss_fn = lambda output: output.mean() if isinstance(output, torch.Tensor) else None

    core_audit = run_model_audit(
        model,
        sample_batch=sample_batch,
        auto_probe=False,
        include_gradients=bool(args.include_gradients),
        loss_fn=loss_fn,
    )
    hidden = _scan_hidden_elements(model)
    parameter_space = _parameter_space_breakdown(model)

    payload = {
        "metadata": {
            "generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_name": args.model,
            "probe": {
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "input_dim": int(args.input_dim),
                "include_gradients": bool(args.include_gradients),
                "seed": int(args.seed),
            },
        },
        "audit": core_audit,
        "parameter_space": parameter_space,
        "hidden_elements": hidden,
    }
    markdown = _make_report_markdown(payload)

    json_path = os.path.join(ROOT, args.output_json)
    md_path = os.path.join(ROOT, args.output_md)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote JSON audit to: {json_path}")
    print(f"Wrote Markdown report to: {md_path}")
    print(f"Total parameters: {core_audit['total_parameters']:,}")
    print(f"Active modules in probe: {core_audit['utilization']['active_modules']}/{core_audit['utilization']['total_modules']}")
    print(f"Hidden modules found: {hidden['totals']['hidden_module_count']}")
    print(f"Hidden tensors found: {hidden['totals']['hidden_tensor_count']}")


if __name__ == "__main__":
    main()
