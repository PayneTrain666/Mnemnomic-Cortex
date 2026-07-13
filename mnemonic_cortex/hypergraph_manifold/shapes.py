"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: shapes.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Tensor shape contracts for hyperset probability matrices/tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple
import math

from .validation import ValidationResult


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(float(value))


def nested_shape(data: Any) -> Tuple[int, ...]:
    """Return uniform nested-list shape.

    Scalars have shape ``()``. Ragged sequences raise ``ValueError``.
    Objects exposing a ``shape`` attribute are accepted, but the actual
    values cannot be validated unless they are nested Python sequences.
    """

    if hasattr(data, "shape") and not isinstance(data, (list, tuple)):
        return tuple(int(x) for x in data.shape)
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        child_shapes = [nested_shape(x) for x in data]
        first = child_shapes[0]
        if any(shape != first for shape in child_shapes):
            raise ValueError("ragged nested probability data is not allowed")
        return (len(data),) + first
    return ()


def iter_leaf_values(data: Any, prefix: Tuple[int, ...] = ()) -> Iterator[Tuple[Tuple[int, ...], Any]]:
    if isinstance(data, (list, tuple)):
        for idx, child in enumerate(data):
            yield from iter_leaf_values(child, prefix + (idx,))
    else:
        yield prefix, data


@dataclass(frozen=True)
class TensorShapeContract:
    name: str
    axes: Tuple[str, ...]
    description: str = ""

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def mutation_axis_index(self) -> int:
        if "m" not in self.axes:
            raise ValueError(f"Contract {self.name} has no mutation axis 'm'")
        return self.axes.index("m")

    def validate_shape(self, shape: Sequence[int]) -> ValidationResult:
        result = ValidationResult()
        shape_tuple = tuple(int(x) for x in shape)
        if len(shape_tuple) != self.rank:
            result.error(
                "shape.rank_mismatch",
                f"{self.name} expects rank {self.rank} with axes {self.axes}, got shape {shape_tuple}",
                path="probabilities",
            )
        for axis, size in zip(self.axes, shape_tuple):
            if size <= 0:
                result.error(
                    "shape.non_positive_axis",
                    f"Axis {axis!r} must be positive, got {size}",
                    path=f"shape.{axis}",
                )
        return result


P_VM = TensorShapeContract("P[v,m]", ("v", "m"), "Variable by mutation-magnitude matrix")
P_VMD = TensorShapeContract("P[v,m,d]", ("v", "m", "d"), "Variable/mutation/depth tensor")
P_VMDC = TensorShapeContract("P[v,m,d,c]", ("v", "m", "d", "c"), "Adds context clusters")
P_VMDCT = TensorShapeContract("P[v,m,d,c,t]", ("v", "m", "d", "c", "t"), "Adds time")
P_VMDCTA = TensorShapeContract("P[v,m,d,c,t,a]", ("v", "m", "d", "c", "t", "a"), "Adds action candidate")

SHAPE_CONTRACTS: Dict[str, TensorShapeContract] = {
    item.name: item
    for item in (P_VM, P_VMD, P_VMDC, P_VMDCT, P_VMDCTA)
}


def contract_by_name(name: str) -> TensorShapeContract:
    try:
        return SHAPE_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tensor shape contract: {name!r}") from exc
