from pathlib import Path

from mnemonic_cortex.hypergraph_manifold import (
    HGM10ReleaseOptions,
    build_hgm10_release_consolidation,
    build_hgm_integration_roadmap,
    consolidate_hgm_release_manifests,
    freeze_hgm_public_api,
)


def test_api_freeze_records_public_symbols():
    result = freeze_hgm_public_api(options=HGM10ReleaseOptions(max_symbols=512))
    assert result.frozen is True
    assert result.symbols
    names = {symbol.symbol_name for symbol in result.symbols}
    assert "MutationToken" in names
    assert "build_hgm9_runtime_integration_evaluation" in names
    assert "build_hgm10_release_consolidation" in names
    assert all(symbol.exported for symbol in result.symbols)
    assert result.metadata["live_qdt_write"] is False


def test_api_freeze_is_deterministic_for_ids():
    a = freeze_hgm_public_api(options=HGM10ReleaseOptions(max_symbols=512))
    b = freeze_hgm_public_api(options=HGM10ReleaseOptions(max_symbols=512))
    assert a.freeze_id == b.freeze_id
    assert [s.symbol_id for s in a.symbols] == [s.symbol_id for s in b.symbols]


def test_api_freeze_symbol_bound_is_enforced():
    result = freeze_hgm_public_api(options=HGM10ReleaseOptions(max_symbols=5))
    assert len(result.symbols) == 5
    assert result.validation.warnings


def test_release_consolidation_collects_manifests_and_docs():
    root = Path.cwd()
    result = consolidate_hgm_release_manifests(root)
    assert result.manifest_summaries
    assert any(summary.stage_id == "hgm_9" for summary in result.manifest_summaries)
    assert result.documentation_files
    assert any(path.endswith("10_hgm_9_runtime_integration_readiness.md") for path in result.documentation_files)
    assert result.metadata["live_qdt_write"] is False


def test_release_consolidation_missing_root_degrades_safely(tmp_path):
    result = consolidate_hgm_release_manifests(tmp_path)
    assert result.validation.ok or result.validation.warnings
    assert result.manifest_summaries == tuple()
    assert result.metadata["found_stage_count"] == 0


def test_integration_roadmap_has_finalize_command():
    result = build_hgm_integration_roadmap()
    assert result.items
    assert result.next_command.startswith("DEV-FLOW FINALIZE HGM-V0.1")
    assert any(item.stage == "HGM-V0.1-FINALIZE" for item in result.items)
    assert result.metadata["production_enabled"] is False


def test_integration_roadmap_item_bound_is_enforced():
    result = build_hgm_integration_roadmap(options=HGM10ReleaseOptions(max_roadmap_items=2))
    assert len(result.items) == 2
    assert result.validation.warnings


def test_high_level_hgm10_consolidation_builds_all_records():
    result = build_hgm10_release_consolidation(Path.cwd())
    assert result.api_freeze.frozen is True
    assert result.release_consolidation.manifest_summaries
    assert result.integration_roadmap.next_command.startswith("DEV-FLOW FINALIZE")
    assert result.metadata["additive_only"] is True
    assert result.metadata["live_qdt_write"] is False


def test_hgm10_options_validate_positive_bounds():
    try:
        HGM10ReleaseOptions(max_symbols=0)
    except ValueError as exc:
        assert "max_symbols" in str(exc)
    else:
        raise AssertionError("expected max_symbols validation failure")


def test_trace_records_are_generated_and_redacted():
    result = build_hgm10_release_consolidation(Path.cwd())
    assert result.trace_records
    rendered = str([trace.redacted_payload() for trace in result.trace_records])
    assert "must_redact" not in rendered
