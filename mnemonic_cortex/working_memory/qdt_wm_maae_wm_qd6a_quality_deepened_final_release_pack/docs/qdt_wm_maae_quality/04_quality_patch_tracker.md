# WM-QD Quality Patch Tracker

| tracker_id | stage | affected files | reason | severity | status | required action | downstream impact | completion evidence |
|---|---|---|---|---|---|---|---|---|
| WM-QD-0A-0001 | WM-QD-0A | quality package | Quality tooling missing | blocker | complete | Create schemas/classifier/planner/lineage/report | Enables WM-QD-1A | Source/tests generated |
| WM-QD-0A-0002 | WM-QD-0A | quality docs | Quality docs missing | high | complete | Create docs and command library | Enables controlled hardening | Docs generated |
| WM-QD-0A-0003 | WM-QD-0A | real cortex source | Real source absent | medium | partially_complete | Keep deferred until source supplied | Production integration still requires real patch | Deferred honestly |

| WM-QD-0A-PATCH-0001 | WM-QD-0A | wm_quality_issue_schema.py | Direct issue construction lacked default safety payload | blocker | complete | Auto-fill no-mutation safety payload in __post_init__ | Required for serialization safety | Full tests rerun |

| WM-QD-1A-0001 | WM-QD-1A | wm_foundation_guards.py | Shared early-WM validation helpers missing | high | complete | Add finite/shape/serialization/trace guards | Enables consistent foundation hardening | Source/tests produced |
| WM-QD-1A-0002 | WM-QD-1A | early WM modules | Explicit module-level quality contracts missing | high | complete | Add wm_qd1a_foundation_contract to each present early module | Enables bounded classifier/remediation use | Contract test passes |
| WM-QD-1A-0003 | WM-QD-1A | tests/test_wm_qd1a_* | Foundation hardening tests missing | high | complete | Add guard/contract/classifier tests | Prevents fake hardening | Full tests pass |
| WM-QD-1A-0004 | WM-QD-1A | missing early files | Any absent historical early files must be reported | medium | complete | Record missing_scope_files in docs | Avoids fake source claims | Docs generated |

| WM-QD-2A-0001 | WM-QD-2A | wm_depth_guards.py | Shared quaternion-depth validation helpers missing | high | complete | Add depth/triplet/quaternion guards | Enables consistent depth hardening | Source/tests produced |
| WM-QD-2A-0002 | WM-QD-2A | depth/assembly modules | Explicit module-level depth contracts missing | high | complete | Add wm_qd2a_depth_contract to each present in-scope module | Enables bounded classifier/remediation use | Contract test passes |
| WM-QD-2A-0003 | WM-QD-2A | tests/test_wm_qd2a_* | Depth hardening tests missing | high | complete | Add guard/contract/QDT regression/classifier tests | Prevents fake hardening | Full tests pass |
| WM-QD-2A-0004 | WM-QD-2A | missing depth files | Any absent depth files must be reported | medium | complete | Record missing_scope_files in docs | Avoids fake source claims | Docs generated |

| WM-QD-2A-PATCH-0001 | WM-QD-2A | wm_depth_guards.py | Depth validators leaked WMFoundationValidationError instead of WMDepthValidationError | blocker | complete | Normalize depth/token/quaternion validation exceptions to WMDepthValidationError | Makes depth contract catchable and consistent | Full tests rerun |

| WM-QD-3A-0001 | WM-QD-3A | wm_attention_guards.py | Shared attention validation helpers missing | high | complete | Add query/candidate/score/lane/trace guards | Enables consistent attention hardening | Source/tests produced |
| WM-QD-3A-0002 | WM-QD-3A | attention modules | Explicit module-level attention contracts missing | high | complete | Add wm_qd3a_attention_contract to each present in-scope module | Enables bounded classifier/remediation use | Contract test passes |
| WM-QD-3A-0003 | WM-QD-3A | tests/test_wm_qd3a_* | Attention hardening tests missing | high | complete | Add guard/contract/QDT regression/classifier tests | Prevents fake hardening | Full tests pass |
| WM-QD-3A-0004 | WM-QD-3A | missing attention files | Any absent attention files must be reported | medium | complete | Record missing_scope_files in docs | Avoids fake source claims | Docs generated |

| WM-QD-4A-0001 | WM-QD-4A | wm_external_memory_guards.py | Shared external-memory/shared/QH validation helpers missing | high | complete | Add response/MANN/fusion/shared-slot/QH/interference guards | Enables consistent external-memory hardening | Source/tests produced |
| WM-QD-4A-0002 | WM-QD-4A | external/shared/QH modules | Explicit module-level external-memory contracts missing | high | complete | Add wm_qd4a_external_memory_contract to each present in-scope module | Enables bounded classifier/remediation use | Contract test passes |
| WM-QD-4A-0003 | WM-QD-4A | tests/test_wm_qd4a_* | External-memory hardening tests missing | high | complete | Add guard/contract/runtime/classifier tests | Prevents fake hardening | Full tests pass |
| WM-QD-4A-0004 | WM-QD-4A | missing external files | Any absent external-memory files must be reported | medium | complete | Record missing_scope_files in docs | Avoids fake source claims | Docs generated |

| WM-QD-5A-0001 | WM-QD-5A | wm_commit_cortex_guards.py | Shared commit/cortex validation helpers missing | high | complete | Add proposal/decision/rollback/wrapper/migration safety guards | Enables consistent commit/cortex hardening | Source/tests produced |
| WM-QD-5A-0002 | WM-QD-5A | commit/cortex modules | Explicit module-level commit/cortex contracts missing | high | complete | Add wm_qd5a_commit_cortex_contract to each present in-scope module | Enables bounded classifier/remediation use | Contract test passes |
| WM-QD-5A-0003 | WM-QD-5A | tests/test_wm_qd5a_* | Commit/cortex hardening tests missing | high | complete | Add guard/contract/runtime/classifier tests | Prevents fake hardening | Full tests pass |
| WM-QD-5A-0004 | WM-QD-5A | real cortex source patch | Real EnhancedMnemonicCortex source still absent | medium | partially_complete | Preserve migration-template-only stance | Avoids fake patch claim | Deferred register updated |

| WM-QD-5A-PATCH-0001 | WM-QD-5A | wm_commit_cortex_guards.py | Commit/cortex validators leaked WMFoundationValidationError instead of WMCommitCortexValidationError | blocker | complete | Normalize proposal/decision validation exceptions to WMCommitCortexValidationError | Makes commit/cortex contract catchable and consistent | Full tests rerun |

| WM-QD-6A-0001 | WM-QD-6A | API/dependency audit | Final source/API/dependency audit required | high | complete | Regenerate API/dependency audits | Supports release closure | Docs generated |
| WM-QD-6A-0002 | WM-QD-6A | WM-QD contracts | All quality contracts must be verified | blocker | complete | Run contract verification script | Ensures WM-QD-1A through WM-QD-5A contracts callable | Contract verification passed |
| WM-QD-6A-0003 | WM-QD-6A | tests/benchmarks | Final tests and smoke benchmarks required | blocker | complete | Run full pytest and benchmark harness | Confirms quality-deepened pack is runnable | Test/benchmark outputs captured |
| WM-QD-6A-0004 | WM-QD-6A | release/readiness docs | Final release manifest/readiness/deferred docs required | high | complete | Generate manifest/readiness/deferred hardening plan | Prevents fake production-complete claim | Docs generated |
| WM-QD-6A-0005 | WM-QD-6A | real cortex source patch | Real EnhancedMnemonicCortex source still absent | medium | partially_complete | Keep honest deferred requirement | Production integration still requires real patch | Deferred plan updated |
