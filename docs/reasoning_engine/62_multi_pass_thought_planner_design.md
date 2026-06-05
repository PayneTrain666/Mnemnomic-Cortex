# REASON-3A Multi-Pass Thought Planner Design

`MultiPassThoughtPlanner` is disabled by default. When enabled, it accepts `[B,D]` or `[B,T,D]` tensors, builds or consumes a strategy graph, expands routes, and produces bounded planning passes.

It does not mutate input tensors, does not write to memory stores, and emits PAAMA-X-compatible trace metadata.
