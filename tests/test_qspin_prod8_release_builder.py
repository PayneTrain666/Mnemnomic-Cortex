from qspin_prod8_module_loader import load_module
m = load_module("qspin_prod8_release_builder")

def test_release_builder_blocks_missing_critical_artifacts():
    builder = m.Prod8ExpandedReleaseBuilder()
    ok = builder.build(m.Prod8ReleaseBuildRequest("ok", m.build_default_prod8_release_artifacts()))
    assert ok.status.value == "ready_to_package"
    items = list(m.build_default_prod8_release_artifacts())
    items[0] = m.Prod8ReleaseArtifactRecord("missing", m.Prod8ReleaseArtifactKind.MODULE, "x", False, True)
    bad = builder.build(m.Prod8ReleaseBuildRequest("bad", tuple(items)))
    assert bad.status.value == "blocked"
