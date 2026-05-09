# WM-QD-1A Acceptance, Audit, and Ship-Check

## Ship-check JSON

```json
{
  "stage": "WM-QD-1A",
  "stage_complete": true,
  "wm_qd_campaign_complete": false,
  "source_files_produced_or_patched": [
    "wm_foundation_guards.py",
    "context_geometry_maps.py",
    "context_map_selector.py",
    "context_to_wm_bridge.py",
    "wm_context_mount.py",
    "legacy_enhanced_curved_memory.py",
    "wm_curved_core.py",
    "curved_resonant_wm_core.py",
    "curved_slot_state.py",
    "curvature_metric_policy.py",
    "geometry_aware_addressing.py",
    "bounded_associative_spread.py",
    "curved_local_trace.py",
    "curved_shadow_write.py"
  ],
  "tests_produced": [
    "tests/test_wm_qd1a_foundation_guards.py",
    "tests/test_wm_qd1a_module_contracts.py",
    "tests/test_wm_qd1a_quality_classifier_scope.py"
  ],
  "docs_produced": [
    "docs/qdt_wm_maae_quality/09_wm_qd1a_foundation_quality_deepening.md",
    "docs/qdt_wm_maae_quality/10_wm_qd1a_deferred_register.md",
    "docs/qdt_wm_maae_quality/11_exact_wm_qd2a_continuation_command.md",
    "docs/qdt_wm_maae_quality/12_wm_qd1a_pytest_output.txt",
    "docs/qdt_wm_maae_quality/13_wm_qd1a_acceptance_shipcheck.md"
  ],
  "before_hashes": {
    "context_geometry_maps.py": "45c2de359e7b0e1096ac4fe45f926d6efbf771e50e85cc99de368331c96eb0b6",
    "context_map_selector.py": "3d95163ff82d06f748d16ba2adce3e23502206e256b2dbc354431f43ac01d2ea",
    "context_to_wm_bridge.py": "b3b72255a811528f91d30fad547fa76cf14c3a257afba9c26ee8a9ece416be95",
    "wm_context_mount.py": "f39dc1b52d7b44bea5b4c7e54692f26ebb1939b99806a550035f3a6b6a10fa96",
    "legacy_enhanced_curved_memory.py": "5c6c6b874894fdd402be3dc07cdeaa6d2e443a44640b2e03fb7489bb79975f8a",
    "wm_curved_core.py": "a2ad68f47fa339981385355d03571178ee6fbe64af4d2054249efce03e5b8c7a",
    "curved_resonant_wm_core.py": "4d1d2c9a08a26ab4337520b4182b69e5d31c5ba70fe476151140a6b1e99e3e10",
    "curved_slot_state.py": "842f234b4cf77ac5830689fd1de4ada7a4d2c1109a5a02ec612862daa5c86cd8",
    "curvature_metric_policy.py": "a0697f314a61d39a8e78f98ebe56e6148bf79cc43477b8ed3d88c9646a49ce32",
    "geometry_aware_addressing.py": "25b5462e0ac2490b8b49c7918745aaacc127c9bfead99e3b526c821f58d71469",
    "bounded_associative_spread.py": "763c02bab9088ed4e655346da697290fcdf6853f4081c53640eab580710bc936",
    "curved_local_trace.py": "d4c342d36a521c7c3d6b3323f67c2b7258f943895331bb0013c67a1d5460eefb",
    "curved_shadow_write.py": "89c90258e4e53b8b702f3bac6ec4edb0283224bb1d891652c0dc8daf4aef2ee7"
  },
  "after_hashes": {
    "context_geometry_maps.py": "0b341a74a347ceda7afa8f3bd53284b7aefb0947c4906371e47bcb403a007cc9",
    "context_map_selector.py": "ec807cd65ce3c0d1c78e8c7e492bf8b0269b6d183e0c4a9e4863b38cc4a52eda",
    "context_to_wm_bridge.py": "d857f56ca1b6a8b0e39b39357e5a12ed29602d7474e321b445d33ba043957586",
    "wm_context_mount.py": "40d0aedc98cf97509d024c21728bbf92b698d42fb0dc15fba691bd24d4902b75",
    "legacy_enhanced_curved_memory.py": "f6c4d2e3a727cebee7e3488e400f14205556faf8e47f28ba0369d977061b28e0",
    "wm_curved_core.py": "b63ab3b6531ab07a46bb6c3023c0c119b0f025c9d6976881c0b04d85336a3dcf",
    "curved_resonant_wm_core.py": "ba0f8afa89187290076ddfd590571bded43303eb09a883add9958a10e61cc9b6",
    "curved_slot_state.py": "c72acd19ab020116b211be7118ae9a69b57c3e799c44feede11cb0e2811e886e",
    "curvature_metric_policy.py": "062c3738e6879456c41bf79b33066f5192387ba44a3a96cf5cb1d1cacf62620d",
    "geometry_aware_addressing.py": "7b2038bceaa13a949b9b849da5e124c364f80f5f9356f6919defaa30cea3c8ff",
    "bounded_associative_spread.py": "a4bb73ec8c53e800026c5e95857647fab74496904fd8d9b04824b8d5009c25ec",
    "curved_local_trace.py": "65f92e05a58e6f764a8ace9af5170cf97097c8a77d88e1bfe1a2b97cc40ef60f",
    "curved_shadow_write.py": "0699f6d4971b3f423f9699762d6f4ef377068fcfd84da485da91a55e0348c94d"
  },
  "missing_scope_files": [],
  "redo_required": false
}
```

## Pytest output

```text
Spreadsheet runtime warmup failed during python startup
Traceback (most recent call last):
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/patches/warm_spreadsheet_runtime_on_startup.py", line 26, in warm_spreadsheet_runtime_on_startup
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 785, in warm_spreadsheet_runtime
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 720, in _warm_feature_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 704, in _warm_collaboration_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/generated/interface/models.py", line 48821, in hydrate_crdt_from_proto
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/remote.py", line 747, in __call__
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/client.py", line 150, in call
artifact_tool.rpc.client.RemoteError: hydrateCrdtFromProto requires an empty collaborative document.
[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m [ 52%]
[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[32m        [100%][0m
[32m[32m[1m137 passed[0m[32m in 6.57s[0m[0m

```

## Full-Depth Adequacy Gate

PASS — WM-QD-1A is deep enough for the early foundation hardening pass.

## Next command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-2A — Quaternion Depth, Depth Transformer, Triplet State, Trace, Depth Fusion, and QDTWorkingMemory Assembly Quality Deepening

Goal:
Use the WM-QD-0A/WM-QD-1A quality tooling to harden the quaternion/depth/assembly layer.

Required:
1. Read the latest WM-QD-1A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_quaternion_depth.py
   - wm_intra_depth_transformer.py
   - wm_cross_depth_transformer.py
   - depth_specific_addressing.py
   - wm_depth_adapters.py
   - wm_depth_fusion.py
   - wm_triplet_state.py
   - wm_trace.py
   - qdt_working_memory.py
3. Strengthen shape checks, finite checks, depth/triplet invariants, quaternion normalization guarantees, trace serialization, fallback behavior, and boundedness.
4. Preserve 8-depth default, triplet representation, true quaternion depth rotation, context-map mounting, curved core, and PAAMA-X metadata.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-3A continuation command.

```
