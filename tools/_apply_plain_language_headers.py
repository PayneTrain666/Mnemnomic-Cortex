#!/usr/bin/env python3
"""
Plain-language summary
----------------------
What this file is for: One-shot maintenance script that inserts plain-language module headers.
How it fits in the system: Documentation tooling used to annotate the codebase for non-coders.
Status: WORKING (maintenance)
Important notes for non-coders: Safe to re-run; skips files that already have the header marker.

Technical notes (original):
One-shot helper: add plain-language module headers to active product files.

Idempotent: skips files that already contain 'Plain-language summary'.
Does not touch tests, pytorch_new, or the QD6A release pack.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_SUBSTR = "qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack"
MARKER = "Plain-language summary"

# Explicit overrides: relative posix path -> (what, fits, status, notes)
OVERRIDES: dict[str, tuple[str, str, str, str]] = {
    "mnemonic_cortex/__init__.py": (
        "Public package entry that exports parameter-storage loop and CMS depth-stack helpers.",
        "This is what `import mnemonic_cortex` exposes first for those advanced storage features.",
        "ACTIVE",
        "Most of the brain lives in cortex.py and subpackages, not only this file.",
    ),
    "mnemonic_cortex/cortex.py": (
        "The top-level brain controller: connects sensory buffer, working memory, long-term memory, consolidation, and safety features.",
        "Almost every training or inference path eventually goes through EnhancedMnemonicCortex here.",
        "ACTIVE / WORKING",
        "Huge file — look for section banners (INIT / BUILD, FORWARD PATH, etc.). Optional stacks depend on feature flags.",
    ),
    "mnemonic_cortex/triple_hybrid.py": (
        "The long-term memory engine that runs several memory banks and fuses their answers.",
        "Cortex uses this as the main LTM; holographic slot banks (qh_banks) live here too.",
        "ACTIVE / WORKING",
        "qh_banks is a ModuleDict so GPU moves work with model.to(cuda).",
    ),
    "mnemonic_cortex/quantum_holographic.py": (
        "Stores memories as overlapping hologram patterns in slots, with codes for depth/bank/role.",
        "Used by LTM banks and consolidated memory to stack several memories in one address.",
        "WORKING",
        "Codebook tensors move with the module via _apply when you call .to(device).",
    ),
    "mnemonic_cortex/config.py": (
        "Named knobs (sizes, flags, capacities) for a cortex build.",
        "Builders and loaders read these when assembling the model.",
        "ACTIVE",
        "Changing defaults changes how large and expensive a default model is.",
    ),
    "mnemonic_cortex/config_loader.py": (
        "Reads YAML / unified config files into live settings objects.",
        "Bridges human-editable configs to cortex construction.",
        "ACTIVE",
        "Prefer config files + this loader over hard-coding sizes elsewhere.",
    ),
    "mnemonic_cortex/capacity_profile.py": (
        "Preset size envelopes such as compact, standard, and deep.",
        "Quick coherent sets of slot counts and dimensions.",
        "ACTIVE",
        "Deep profiles need more GPU memory.",
    ),
    "mnemonic_cortex/sensory_buffer.py": (
        "Short-term holding area for recent inputs, with attention and salience.",
        "First memory stage before working memory and long-term retrieval.",
        "ACTIVE",
        "Like a small echo of what just happened.",
    ),
    "mnemonic_cortex/consolidated_memory.py": (
        "Consolidated Memory Store (CMS) with holographic slot support.",
        "Used when advanced consolidation / depth stacking is enabled.",
        "ACTIVE when enabled / OPT-IN otherwise",
        "Depth stack lives in consolidated_memory_depth_stack.py.",
    ),
    "mnemonic_cortex/consolidated_memory_depth_stack.py": (
        "Multi-geometry depth stack on top of consolidated memory.",
        "Adds depth and hidden processing when advanced consolidation is on.",
        "OPT-IN / ACTIVE in full-stack builds",
        "Large free-hidden settings cost many parameters.",
    ),
    "mnemonic_cortex/parameter_storage_loop_stack.py": (
        "Optional loop storing parameter bundles across geometry manifolds.",
        "Extra structure beyond ordinary weight tensors.",
        "OPT-IN",
        "Training writes may stay disabled even when present.",
    ),
    "mnemonic_cortex/memory_hg.py": (
        "Hypergeometric / episodic-style long-term memory bank.",
        "One of the main LTM banks inside triple-hybrid fusion.",
        "ACTIVE",
        "Stores episode-like patterns rather than only word meanings.",
    ),
    "mnemonic_cortex/memory_cgmn.py": (
        "CGMN semantic long-term memory bank.",
        "Holds meaning-oriented patterns used during fusion.",
        "ACTIVE",
        "Works alongside HG, curved/SPCP, and spatial banks.",
    ),
    "mnemonic_cortex/memory_curved.py": (
        "Curved / resonant memory bank (also historically tied to older WM).",
        "Provides a curved-geometry memory path inside LTM / legacy WM.",
        "ACTIVE / LEGACY overlap",
        "Modern WM prefers QDT; this bank remains important in LTM.",
    ),
    "mnemonic_cortex/memory_spatial_ltm.py": (
        "Spatial / topological atlas bank for places and structure.",
        "Optional spatial LTM path wired into triple-hybrid.",
        "OPT-IN / ACTIVE when spatial LTM is enabled",
        "Useful when the task needs layout or map-like memory.",
    ),
    "mnemonic_cortex/memory_attention.py": (
        "Multi-scale attention used when reading or writing memory.",
        "Shared attention building block for memory modules.",
        "ACTIVE",
        "Helps the model focus on the most relevant memory pieces.",
    ),
    "mnemonic_cortex/memory_transformer_v2.py": (
        "Transformer wrappers around the individual memory banks.",
        "Lets each bank refine its own representation before fusion.",
        "ACTIVE",
        "Prefer this v2 stack over older transformer helpers.",
    ),
    "mnemonic_cortex/memory_consolidation_manager_v2.py": (
        "Orchestrates consolidation across memory banks.",
        "Coordinates when short-term patterns become longer-term stores.",
        "ACTIVE when consolidation path is on",
        "Works with brokers and CMS helpers.",
    ),
    "mnemonic_cortex/ltm_aux_memory.py": (
        "Helper long-term banks (consolidated LTM bank and neural field memory).",
        "Auxiliary stores beside the main triple-hybrid banks.",
        "ACTIVE / LOW-USE depending on build",
        "Not always on the hottest path.",
    ),
    "mnemonic_cortex/spatial_ltm_cortex_wiring.py": (
        "Wiring glue between spatial LTM and the cortex / WM lattice.",
        "Connects spatial banks so cortex can mirror and use them.",
        "ACTIVE when spatial LTM enabled",
        "Mostly integration code, not a standalone memory.",
    ),
    "mnemonic_cortex/hg_episodic_cortex_wiring.py": (
        "Wiring glue between HG episodic LTM and cortex / WM lattice.",
        "Connects episodic banks into the larger system.",
        "ACTIVE",
        "Integration helpers for HG episodic doctrine.",
    ),
    "mnemonic_cortex/hidden_attention_orchestrator.py": (
        "Collects and routes hidden activations across many modules.",
        "Gives a global view of internal signals for attention / diagnostics.",
        "ACTIVE in full stacks",
        "Important for hidden-attention training tasks.",
    ),
    "mnemonic_cortex/ahg.py": (
        "Anti-Hallucination Guard: decides when an answer looks unsupported.",
        "Safety layer that can block or flag risky outputs.",
        "OPT-IN",
        "Enable when evaluating truthfulness-sensitive tasks.",
    ),
    "mnemonic_cortex/anti_hallucination.py": (
        "Helper math and thresholds used by anti-hallucination checks.",
        "Supports AHG decisions with entropy/margin style signals.",
        "OPT-IN / WORKING",
        "Companion to ahg.py.",
    ),
    "mnemonic_cortex/lightbulb.py": (
        "Detects 'aha' moments and can boost recall temperature.",
        "Triggers stronger memory recall when novelty or importance spikes.",
        "OPT-IN / ACTIVE when enabled",
        "Older lightbulb path; see also lightbulb_recall_v2.",
    ),
    "mnemonic_cortex/lightbulb_recall_v2.py": (
        "Newer lightbulb-style explosive recall controller.",
        "Modernized recall boost path used by some cortex builds.",
        "ACTIVE when enabled (prefer over v1 where wired)",
        "Works with event logging.",
    ),
    "mnemonic_cortex/lightbulb_controller.py": (
        "Tracks running statistics that decide lightbulb activation.",
        "Controller layer above raw lightbulb detectors.",
        "OPT-IN",
        "Tuning here changes how often 'aha' recall fires.",
    ),
    "mnemonic_cortex/lightbulb_event_logger.py": (
        "Records when lightbulb / recall events fire.",
        "Diagnostics and analysis aid for recall behavior.",
        "WORKING (utility)",
        "Does not change memory content by itself.",
    ),
    "mnemonic_cortex/topology_manager.py": (
        "Older topology / manifold warping helpers.",
        "Geometry utilities historically used for memory topology.",
        "LEGACY / still referenced",
        "Prefer topology_manager_v2 where cortex uses it.",
    ),
    "mnemonic_cortex/topology_manager_v2.py": (
        "Newer topology manager with history tracking.",
        "Helps cortex navigate geometry choices over time.",
        "ACTIVE",
        "Preferred over topology_manager.py for new wiring.",
    ),
    "mnemonic_cortex/hybrid_router_v2.py": (
        "Routes information among hybrid memory banks.",
        "Decides how much each bank should contribute.",
        "ACTIVE",
        "Central traffic controller inside triple-hybrid style stacks.",
    ),
    "mnemonic_cortex/router_advanced.py": (
        "Advanced domain router across memory / skill domains.",
        "Higher-level routing used with multi-CPS / distillation setups.",
        "ACTIVE / OPT-IN by feature",
        "Related losses live in router_losses.py.",
    ),
    "mnemonic_cortex/router_losses.py": (
        "Training losses that keep routers well-behaved.",
        "Regularizes routing decisions during learning.",
        "ACTIVE when advanced routing trains",
        "Not a runtime memory store.",
    ),
    "mnemonic_cortex/consolidated_lexicon.py": (
        "Shared vocabulary / concept lexicon backed by geometric distances.",
        "Lets consolidation attach stable names/concepts to patterns.",
        "ACTIVE when lexicon/CMS path on",
        "Works with CMS ops and CPS bridges.",
    ),
    "mnemonic_cortex/cms_ops.py": (
        "Operations for CMS logging, EMA consolidation, and shard dump/load.",
        "Practical tools that move consolidated knowledge in and out.",
        "ACTIVE when CMS path on",
        "Includes safety clamps for geometry.",
    ),
    "mnemonic_cortex/cms_index.py": (
        "Index structures for looking up consolidated memory entries.",
        "Speeds or organizes CMS addressing.",
        "ACTIVE when CMS path on",
        "Supporting structure, not the full store.",
    ),
    "mnemonic_cortex/consolidation_broker.py": (
        "Older consolidation broker that coordinates stores.",
        "Legacy orchestration for consolidation jobs.",
        "LEGACY / still referenced",
        "Prefer consolidation_broker_v2 where cortex uses it.",
    ),
    "mnemonic_cortex/consolidation_broker_v2.py": (
        "Newer consolidation broker with clearer config.",
        "Coordinates multi-store consolidation in modern builds.",
        "ACTIVE when consolidation enabled",
        "Preferred broker implementation.",
    ),
    "mnemonic_cortex/consolidation_scheduler.py": (
        "Schedules when consolidation should run.",
        "Timing layer above brokers.",
        "OPT-IN / ACTIVE when scheduled consolidation used",
        "Does not store memories itself.",
    ),
    "mnemonic_cortex/cps.py": (
        "Consolidated Parameter Store: shared parameter / concept geometry store.",
        "Holds unified parameters that multiple domains can share.",
        "ACTIVE / OPT-IN by feature",
        "Often paired with CPS fuser and multi-CPS manager.",
    ),
    "mnemonic_cortex/cps_fuser.py": (
        "Fuses multiple CPS views into one usable signal.",
        "Combines parameter-store outputs for downstream use.",
        "ACTIVE when CPS path on",
        "Quant-aware cousin exists in quant_fuser.py.",
    ),
    "mnemonic_cortex/cps_vocab_bridge.py": (
        "Bridges CPS keys to vocabulary tokens / skills / entities.",
        "Connects parameter-store identities to language-like tokens.",
        "ACTIVE when CPS lexicon bridging used",
        "Mostly naming and key helpers.",
    ),
    "mnemonic_cortex/multi_cps.py": (
        "Manages several Consolidated Parameter Stores at once.",
        "Multi-domain CPS orchestration.",
        "OPT-IN",
        "Some related tests have historically been flaky — check health register.",
    ),
    "mnemonic_cortex/candidate_view_builder.py": (
        "Builds candidate memory 'views' for comparison or selection.",
        "Turns memory contents into comparable candidate packages.",
        "ACTIVE / WORKING",
        "Used when the system must pick among memory candidates.",
    ),
    "mnemonic_cortex/conflict_resolver.py": (
        "Resolves conflicting memory candidates.",
        "Chooses or blends when memories disagree.",
        "ACTIVE / OPT-IN",
        "Important for coherent recall under conflict.",
    ),
    "mnemonic_cortex/diagnostics.py": (
        "Runtime diagnostics snapshot helpers for the model.",
        "Developer visibility into health and internal stats.",
        "WORKING (utility)",
        "Does not change model behavior by itself.",
    ),
    "mnemonic_cortex/model_audit.py": (
        "Audits which modules fire and how large parameters are.",
        "Produces structural / activation audit reports.",
        "WORKING (utility)",
        "Used by tools/comprehensive_model_audit.py.",
    ),
    "mnemonic_cortex/parameter_audit.py": (
        "Logs parameter-space measurements during runs.",
        "Companion auditing for training or probes.",
        "WORKING (utility)",
        "Developer-facing; safe to ignore for casual use.",
    ),
    "mnemonic_cortex/optimizer.py": (
        "Builds optimizers and learning-rate schedules for training.",
        "Training support, not a memory system.",
        "WORKING",
        "Used by smoke training and some tool scripts.",
    ),
    "mnemonic_cortex/utils.py": (
        "Small utilities: seeding, tensor-core hints, memory-access helpers.",
        "Shared housekeeping for experiments.",
        "WORKING",
        "No model architecture here.",
    ),
    "mnemonic_cortex/train_smoke.py": (
        "Tiny smoke training loop to verify the stack runs.",
        "Quick sanity check rather than full training.",
        "WORKING / LOW-USE",
        "For real GPU curricula prefer tools/copy_task_gpu_train.py.",
    ),
    "mnemonic_cortex/holo_head.py": (
        "Holographic readout head for producing outputs from hologram-like states.",
        "Optional output pathway tied to holographic representations.",
        "OPT-IN / LOW-USE depending on build",
        "Not always the main task decoder.",
    ),
    "mnemonic_cortex/quantization.py": (
        "Int8-style quantization helpers for CPS values.",
        "Compresses or discretizes parameters for efficiency experiments.",
        "OPT-IN",
        "Some related tests have failed in recent full suites — triage if enabling.",
    ),
    "mnemonic_cortex/quant_fuser.py": (
        "Quantization-aware CPS fusion.",
        "Fuses CPS signals while respecting quantized formats.",
        "OPT-IN",
        "Pairs with quantization.py.",
    ),
    "mnemonic_cortex/distillation.py": (
        "Knowledge-distillation helpers across domains.",
        "Lets one part of the system teach another.",
        "OPT-IN",
        "Training technique, not a memory bank.",
    ),
    "mnemonic_cortex/geometry_utils.py": (
        "Low-level geometry helpers (quaternions, angles, etc.).",
        "Math toolkit used by topology and manifold code.",
        "WORKING",
        "Foundation utilities; not a full memory system.",
    ),
    "mnemonic_cortex/geometry_merger.py": (
        "Merges geometry / manifold representations.",
        "Combines geometric views when multiple spaces are active.",
        "ACTIVE / OPT-IN",
        "Used where multi-geometry fusion is required.",
    ),
    "tools/copy_task_gpu_train.py": (
        "Full GPU trainer for copy / reverse sequence tasks with curriculum and checkpoints.",
        "Primary practical training script for the cortex stack on GPU.",
        "WORKING",
        "Keeps a defensive QH device helper; primary fix is in LTM/QH modules.",
    ),
    "tools/comprehensive_model_audit.py": (
        "Writes comprehensive layer and parameter audit reports.",
        "Operator tool for inspecting what the model contains and what activates.",
        "WORKING",
        "Outputs under reports/.",
    ),
    "tools/count_model_params.py": (
        "Prints trainable parameter counts for common copy-task configs.",
        "Quick capacity check before training.",
        "WORKING",
        "Useful when deciding if you have VRAM headroom.",
    ),
    "tools/run_hidden_attention_task.py": (
        "Trains or evaluates the hidden-attention stress task.",
        "Exercises the hidden-attention orchestrator under copy-like workloads.",
        "WORKING",
        "May still call a defensive QH move helper for older paths.",
    ),
    "tools/babi_train_eval.py": (
        "Trains or evaluates on facebook/babi_qa style question answering.",
        "Language reasoning benchmark path for CortexSeqModel.",
        "WORKING (needs dataset)",
        "Depends on external bAbI data availability.",
    ),
    "tools/eval_ahg.py": (
        "Evaluates the Anti-Hallucination Guard on TruthfulQA / FEVER style sets.",
        "Measures whether the guard improves honesty / grounding.",
        "WORKING",
        "Uses eval_adapters.py for model generate contracts.",
    ),
    "tools/eval_adapters.py": (
        "Adapters so evaluation scripts can call different model backends uniformly.",
        "Glue for eval_ahg and similar tools.",
        "WORKING",
        "Not a training script.",
    ),
    "tools/cms_train_loop.py": (
        "Minimal training-loop sketch with CMS logging and consolidation EMA.",
        "Learning aid / sketch more than production trainer.",
        "LOW-USE / sketch",
        "Prefer copy_task_gpu_train.py for serious GPU runs.",
    ),
    "tools/profile.py": (
        "Benchmarks forward/backward step timing for benchmark models.",
        "Performance measurement utility.",
        "WORKING",
        "Does not train to convergence.",
    ),
    "tools/_apply_plain_language_headers.py": (
        "One-shot maintenance script that inserts plain-language module headers.",
        "Documentation tooling used to annotate the codebase for non-coders.",
        "WORKING (maintenance)",
        "Safe to re-run; skips files that already have the header marker.",
    ),
    "tools/_fix_header_future_order.py": (
        "Fixes header placement so the plain-language summary sits before future imports.",
        "Maintenance helper for documentation correctness.",
        "WORKING (maintenance)",
        "Needed because Python module docs must be the first statement.",
    ),
    "tools/_refresh_wm_headers.py": (
        "Rewrites key working-memory file headers with more specific status text.",
        "Maintenance helper used once overrides were added after the first header pass.",
        "WORKING (maintenance)",
        "Only touches a short list of important WM files.",
    ),
}


def _header(what: str, fits: str, status: str, notes: str) -> str:
    return (
        '"""\n'
        f"{MARKER}\n"
        "----------------------\n"
        f"What this file is for: {what}\n"
        f"How it fits in the system: {fits}\n"
        f"Status: {status}\n"
        f"Important notes for non-coders: {notes}\n"
        '"""\n'
    )


def _guess(rel: str) -> tuple[str, str, str, str]:
    if rel in OVERRIDES:
        return OVERRIDES[rel]
    name = Path(rel).name
    stem = Path(rel).stem
    parts = Path(rel).parts

    # Key WM runtime files
    wm_key = {
        "qdt_working_memory.py": (
            "Main modern working-memory assembly (QDT-WM): depths, fusion, and commit-gated writes.",
            "Active scratchpad between sensory input and long-term memory when QDT is enabled.",
            "ACTIVE when QDT-WM enabled",
            "This is the center of the working_memory package.",
        ),
        "wm_config.py": (
            "Configuration objects for QDT working memory.",
            "Knobs that size and enable WM features.",
            "ACTIVE",
            "Change here to resize WM behavior.",
        ),
        "wm_cortex_integration.py": (
            "Hooks that attach QDT working memory into EnhancedMnemonicCortex.",
            "Integration glue between cortex and WM.",
            "ACTIVE when QDT-WM enabled",
            "Not a standalone memory algorithm.",
        ),
        "wm_compatibility_wrapper.py": (
            "Compatibility wrapper so older cortex calls can use QDT-WM.",
            "Bridge between legacy call shapes and the new WM.",
            "ACTIVE / LEGACY bridge",
            "Exists so migration does not break older paths.",
        ),
        "wm_triple_hybrid_ltm_adapter.py": (
            "Adapter that lets working memory read/write the triple-hybrid LTM banks.",
            "Cross-link between WM and LTM fusion.",
            "WORKING",
            "top_k selection always keeps the fused bank.",
        ),
        "wm_quantum_holographic_storage.py": (
            "WM-side quantum holographic storage interface.",
            "Metadata/compatible QH hooks for working memory writes.",
            "INCOMPLETE / interface-compatible",
            "Persistent QH backend is still deferred per readiness docs.",
        ),
        "legacy_enhanced_curved_memory.py": (
            "Older curved working-memory implementation kept for compatibility.",
            "Legacy WM path retained beside QDT-WM.",
            "LEGACY",
            "Can still consume parameters if cortex keeps legacy_working_memory.",
        ),
        "qspin_experimental_live_activation.py": (
            "Experimental controller for guarded QSPIN live activation.",
            "Only relevant if a future stage explicitly authorizes activation.",
            "INERT / guarded",
            "Default remains disabled per project safety rules.",
        ),
    }
    if name in wm_key and "working_memory" in rel.replace("\\", "/"):
        return wm_key[name]

    # QSPIN
    if name.startswith("qspin_") or "/qspin_" in rel.replace("\\", "/"):
        return (
            "QSPIN bridge contract, gate, sandbox, or observability helper.",
            "Documents and guards a future optional bridge; not part of normal live memory routing today.",
            "INERT",
            "Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.",
        )

    # Shim packages
    if parts[0] == "mnemonic_cortex" and parts[1] in {
        "core",
        "consolidation",
        "routing",
        "quant",
        "fusion",
        "distill",
        "geometry",
    }:
        return (
            f"Compatibility import shim for older paths under mnemonic_cortex/{parts[1]}/.",
            "Re-exports the real implementation from elsewhere so old imports keep working.",
            "LEGACY / SHIM",
            "Prefer importing the real module at package root or the named implementation file.",
        )

    if "legacy" in stem:
        return (
            f"Legacy compatibility module ({stem}).",
            "Kept so older setups and rollbacks still work beside newer designs.",
            "LEGACY",
            "Prefer the modern QDT / current modules for new work.",
        )

    if "dry_run" in stem or "dryrun" in stem or "sandbox" in stem:
        return (
            f"Dry-run or sandbox helper ({stem}).",
            "Lets engineers rehearse a path safely without committing live side effects.",
            "LOW-USE / SAFETY SCAFFOLD",
            "Not the everyday training path.",
        )

    if parts[1:2] == ("working_memory",):
        return (
            f"Working-memory (QDT-WM) component: {stem.replace('_', ' ')}.",
            "Part of the active scratchpad stack that sits between sensory input and long-term memory.",
            "ACTIVE / OPT-IN depending on flags",
            "See qdt_working_memory.py for the main assembly; this file is one piece of that stack.",
        )

    if parts[1:2] == ("reasoning_depth",):
        return (
            f"Reasoning-depth component: {stem.replace('_', ' ')}.",
            "Supports multi-layer deeper routing across memory depths when enabled.",
            "OPT-IN",
            "Many adapters stay off until a controller explicitly enables them.",
        )

    if parts[1:2] == ("ltm",):
        return (
            f"Long-term memory package module: {stem.replace('_', ' ')}.",
            "Supports LTM banks, MANN/geometry helpers, or package wiring used with cortex LTM.",
            "ACTIVE / LEGACY depending on file",
            "Some files are local copies or aliases; prefer top-level cortex + triple_hybrid for product runtime.",
        )

    if parts[1:2] == ("memory",):
        return (
            f"Shared-slot memory subsystem module: {stem.replace('_', ' ')}.",
            "Manages shared memory slots that multiple systems can read/write under rules.",
            "OPT-IN",
            "Not always enabled in standard capacity profiles.",
        )

    if parts[1:2] == ("hypergraph_manifold",):
        return (
            f"Hypergraph / HGM manifold module: {stem.replace('_', ' ')}.",
            "Scaffolding for hypergraph probability / procedural manifold routing and write preparation.",
            "LOW-USE / SCAFFOLD (varies)",
            "Many modules are stage artifacts or guarded write-prep rather than the default forward path.",
        )

    if parts[1:2] == ("train",):
        return (
            f"Training helper or placeholder: {stem.replace('_', ' ')}.",
            "Supports package-local training experiments.",
            "LOW-USE",
            "Main GPU training scripts live under tools/.",
        )

    if parts[0] == "tools":
        return (
            f"Project tool script: {stem.replace('_', ' ')}.",
            "Operator / engineer utility for training, evaluation, profiling, or auditing.",
            "WORKING",
            "Run from the repo root with the project environment activated.",
        )

    # Top-level mnemonic_cortex heuristics
    readable = stem.replace("_", " ")
    status = "ACTIVE"
    notes = "See docs/codebase_guide_and_health_audit.md for the map of the whole system."
    if stem.endswith("_v2"):
        status = "ACTIVE (prefer over older v1 twin if both exist)"
    if "audit" in stem or "diagnostic" in stem:
        status = "WORKING (developer utility)"
    if stem in {"topology_manager", "consolidation_broker", "lightbulb"}:
        status = "LEGACY / still referenced — prefer v2 where cortex uses it"
    return (
        f"Product module: {readable}.",
        "Part of the top-level mnemonic_cortex package used by the main cortex build.",
        status,
        notes,
    )


def _existing_docstring(source: str) -> tuple[str | None, int, int] | None:
    """Return (doc, start_offset, end_offset) for module docstring if present."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    node = tree.body[0]
    if not isinstance(node, ast.Expr):
        return None
    val = node.value
    if isinstance(val, ast.Constant) and isinstance(val.value, str):
        return val.value, node.lineno, node.end_lineno or node.lineno
    return None


def _strip_old_doc(source: str) -> str:
    info = _existing_docstring(source)
    if not info:
        return source
    _, start_line, end_line = info
    lines = source.splitlines(keepends=True)
    # Drop docstring lines (1-indexed)
    new_lines = lines[: start_line - 1] + lines[end_line:]
    # Also drop a following blank line if present
    if new_lines and start_line - 1 < len(new_lines) and new_lines[start_line - 1].strip() == "":
        del new_lines[start_line - 1]
    return "".join(new_lines)


def _merge_technical(old_doc: str | None, header: str) -> str:
    if not old_doc:
        return header
    old = old_doc.strip()
    if MARKER in old:
        return '"""\n' + old + '\n"""\n'
    # Keep old technical docstring after the plain-language block
    return header.rstrip()[:-3].rstrip() + "\n\nTechnical notes (original):\n" + old + '\n"""\n'


def process_file(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "skip"

    what, fits, status, notes = _guess(rel)
    header = _header(what, fits, status, notes)

    info = _existing_docstring(text)
    old_doc = info[0] if info else None

    # Preserve coding/future imports before docstring if file starts with them
    # Strategy: if module docstring exists, replace it with merged header.
    # Else insert after shebang / encoding / future imports.
    if info:
        merged = _merge_technical(old_doc, header)
        lines = text.splitlines(keepends=True)
        _, start_line, end_line = info
        new_text = "".join(lines[: start_line - 1]) + merged + "".join(lines[end_line:])
        # ensure blank line after docstring
        if not new_text[len("".join(lines[: start_line - 1])) + len(merged) :].startswith("\n"):
            # ok
            pass
        path.write_text(new_text, encoding="utf-8")
        return "updated-doc"
    else:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        # shebang
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        # encoding cookie
        if insert_at < len(lines) and re.match(r"^#.*coding[:=]", lines[insert_at]):
            insert_at += 1
        # Module docstring must come BEFORE from __future__ imports.
        new_lines = (
            lines[:insert_at]
            + [header if header.endswith("\n") else header + "\n", "\n"]
            + lines[insert_at:]
        )
        path.write_text("".join(new_lines), encoding="utf-8")
        return "inserted"


def main() -> None:
    targets: list[Path] = []
    for p in (ROOT / "mnemonic_cortex").rglob("*.py"):
        if EXCLUDE_SUBSTR in str(p):
            continue
        targets.append(p)
    for p in (ROOT / "tools").glob("*.py"):
        targets.append(p)

    stats = {"skip": 0, "updated-doc": 0, "inserted": 0, "error": 0}
    for path in sorted(targets):
        try:
            result = process_file(path)
            stats[result] = stats.get(result, 0) + 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"ERROR {path}: {exc}")
    print(f"Processed {len(targets)} files: {stats}")


if __name__ == "__main__":
    main()
