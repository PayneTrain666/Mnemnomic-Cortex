"""
Plain-language summary
----------------------
What this file is for: Long-term memory package module: sensory buffer.
How it fits in the system: Supports LTM banks, MANN/geometry helpers, or package wiring used with cortex LTM.
Status: ACTIVE / LEGACY depending on file
Important notes for non-coders: Some files are local copies or aliases; prefer top-level cortex + triple_hybrid for product runtime.

Technical notes (original):
Multimodal sensory ring buffers plus separate context buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class SensoryToken:
    embedding: torch.Tensor
    timestamp: float
    salience: float
    modality: str
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ContextToken:
    embedding: torch.Tensor
    timestamp: float
    kind: str
    priority: float = 1.0
    meta: Dict[str, object] = field(default_factory=dict)


class ModalityRingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[SensoryToken] = []

    def append(self, token: SensoryToken) -> None:
        self.items.append(token)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def read_recent(self, n: int) -> List[SensoryToken]:
        return self.items[-int(n) :]

    def read_salient(self, n: int) -> List[SensoryToken]:
        return sorted(self.items, key=lambda t: float(t.salience), reverse=True)[: int(n)]


class ContextBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[ContextToken] = []

    def append(self, token: ContextToken) -> None:
        self.items.append(token)
        self.items.sort(key=lambda t: (t.priority, t.timestamp), reverse=True)
        self.items = self.items[: self.capacity]

    def snapshot(self, n: Optional[int] = None) -> List[ContextToken]:
        return self.items[: n if n is not None else self.capacity]


class SensoryContextBuffer:
    """Streaming-ready sensory buffers with separate context state."""

    def __init__(self, sensory_capacity: int = 64, context_capacity: int = 32):
        self.sensory_capacity = sensory_capacity
        self.context = ContextBuffer(context_capacity)
        self.modalities: Dict[str, ModalityRingBuffer] = {}

    def add_sensory(self, modality: str, embedding: torch.Tensor, timestamp: float, salience: float = 1.0, **meta) -> None:
        if modality not in self.modalities:
            self.modalities[modality] = ModalityRingBuffer(self.sensory_capacity)
        self.modalities[modality].append(SensoryToken(embedding=embedding.detach(), timestamp=float(timestamp), salience=float(salience), modality=modality, meta=meta))

    def add_context(self, kind: str, embedding: torch.Tensor, timestamp: float, priority: float = 1.0, **meta) -> None:
        self.context.append(ContextToken(embedding=embedding.detach(), timestamp=float(timestamp), kind=kind, priority=float(priority), meta=meta))

    def read(self, mode: str = "now", n: int = 8) -> Dict[str, object]:
        mode = mode.lower()
        if mode == "now":
            return {m: b.read_recent(n) for m, b in self.modalities.items()}
        if mode == "scene":
            return {m: b.read_salient(n) for m, b in self.modalities.items()}
        if mode == "event":
            toks = []
            for b in self.modalities.values():
                toks.extend(b.read_salient(max(1, n // max(1, len(self.modalities)))))
            return {"event_seed": sorted(toks, key=lambda t: t.timestamp)}
        if mode == "spatial":
            spatial = []
            for b in self.modalities.values():
                spatial.extend([t for t in b.items if any(k in t.meta for k in ("pose", "landmark", "motion", "audio_source"))])
            return {"spatial_cues": spatial[-n:]}
        if mode == "semantic":
            return {"semantic_cues": [t for b in self.modalities.values() for t in b.read_salient(n) if "label" in t.meta or "entity" in t.meta]}
        if mode == "episodic":
            return {"context": self.context.snapshot(n), "sensory": self.read("scene", n)}
        raise ValueError(f"unknown sensory read mode: {mode}")

    def snapshot(self) -> Dict[str, int]:
        return {"modalities": len(self.modalities), "context_items": len(self.context.items), **{m: len(b.items) for m, b in self.modalities.items()}}
