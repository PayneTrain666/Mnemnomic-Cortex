from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import ast, re
from .wm_quality_issue_schema import WMQualityEvidence, WMQualityIssue, WMQualityIssueFamily, WMQualityIssueSet, WMQualityLineageRef, WMQualitySeverity
from .wm_quality_lineage import WMQualityLineageIndex, infer_stage

@dataclass
class WMQualityClassifierConfig:
    max_files: int=128; max_issues: int=256; max_file_bytes: int=250_000; require_shape_checks_for_tensor_modules: bool=True; require_trace_hooks_for_wm_modules: bool=True; require_paamax_for_governance_modules: bool=True
    def validate(self):
        if self.max_files<=0 or self.max_issues<=0 or self.max_file_bytes<=0: raise ValueError('bounds must be positive')

class WMQualityClassifier:
    def __init__(self, config: Optional[WMQualityClassifierConfig]=None, lineage_index: Optional[WMQualityLineageIndex]=None):
        self.config=config or WMQualityClassifierConfig(); self.config.validate(); self.lineage_index=lineage_index
    def _lineage(self, file_path: str): return [self.lineage_index.get(file_path)] if self.lineage_index else [WMQualityLineageRef('unknown', infer_stage(file_path), file_path)]
    def _issue(self, family, severity, file_path, summary, evidence_type, snippet=None, line_number=None, metadata=None):
        return WMQualityIssue(family=family,severity=severity,affected_stage=infer_stage(file_path),affected_files=[file_path],summary=summary,evidence=[WMQualityEvidence(evidence_type,summary,snippet,line_number,metadata or {})],lineage=self._lineage(file_path),safety_payload={'classification_only':True,'no_mutation':True,'no_auto_patch_application':True})
    def classify_source_text(self, file_path: str, text: str) -> List[WMQualityIssue]:
        issues=[]; lower=text.lower(); is_py=file_path.endswith('.py'); tensorish='torch' in text or 'tensor' in lower; wm_runtime=file_path.startswith('mnemonic_cortex/working_memory/') and '/quality/' not in file_path
        if is_py and len(text.strip())<120 and not file_path.endswith('__init__.py'): issues.append(self._issue(WMQualityIssueFamily.SHALLOW_SKELETON,WMQualitySeverity.HIGH,file_path,'Python module is very short and may still be a shallow skeleton.','source_text_length',metadata={'length':len(text.strip())}))
        markers=['todo','not implemented','placeholder only','stub']
        if is_py and any(m in lower for m in markers): issues.append(self._issue(WMQualityIssueFamily.SHALLOW_SKELETON,WMQualitySeverity.MEDIUM,file_path,'Source contains placeholder/stub/TODO markers requiring review.','placeholder_marker','; '.join(m for m in markers if m in lower)))
        if is_py and tensorish:
            if self.config.require_shape_checks_for_tensor_modules and not any(m in text for m in ['shape','.dim()','size(','Expected','validate']): issues.append(self._issue(WMQualityIssueFamily.MISSING_SHAPE_CHECK,WMQualitySeverity.HIGH,file_path,'Tensor-using module appears to lack explicit shape validation.','source_text_pattern'))
            if not any(m in text for m in ['torch.isfinite','nan_to_num','NaN','Inf','finite']): issues.append(self._issue(WMQualityIssueFamily.MISSING_FINITE_TENSOR_CHECK,WMQualitySeverity.MEDIUM,file_path,'Tensor-using module appears to lack finite/NaN/Inf checks.','source_text_pattern'))
        if is_py and wm_runtime and self.config.require_trace_hooks_for_wm_modules and not any(m in lower for m in ['trace','to_dict','paamax_metadata']): issues.append(self._issue(WMQualityIssueFamily.MISSING_TRACE_HOOK,WMQualitySeverity.MEDIUM,file_path,'Working-memory runtime module may lack trace/serialization hooks.','source_text_pattern'))
        if is_py and any(n in file_path for n in ['commit','policy','conflict','shared','quantum','holographic','external','dual_fusion']) and self.config.require_paamax_for_governance_modules:
            if 'paamax_metadata' not in lower and 'write_permission' not in lower: issues.append(self._issue(WMQualityIssueFamily.MISSING_PAAMAX_METADATA,WMQualitySeverity.HIGH,file_path,'Governance-sensitive module may lack PAAMA-X/write-permission metadata.','source_text_pattern'))
        if is_py and 'for ' in text and any(m in lower for m in ['rglob','glob(','records.values','all files','while true']) and not any(m in lower for m in ['max_','limit','bounded','cap','top_k']): issues.append(self._issue(WMQualityIssueFamily.UNBOUNDED_SCAN_OR_GROWTH,WMQualitySeverity.MEDIUM,file_path,'Module may perform scans/growth without an obvious bound.','boundedness_pattern'))
        if is_py:
            try: tree=ast.parse(text)
            except SyntaxError as e:
                issues.append(self._issue(WMQualityIssueFamily.UNKNOWN_SIGNATURE,WMQualitySeverity.BLOCKER,file_path,f'Python source failed AST parse: {e}','ast_parse',line_number=e.lineno)); return issues
            for node in tree.body:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and not node.name.startswith('_') and wm_runtime and ast.get_docstring(node) is None and not file_path.startswith('tests/'):
                    issues.append(self._issue(WMQualityIssueFamily.WEAK_TYPE_CONTRACT,WMQualitySeverity.LOW,file_path,f'Public API {node.name} lacks a docstring/contract note.','ast_public_api',line_number=getattr(node,'lineno',None)))
        return issues
    def classify_source_file(self, root: Path, path: Path):
        rel=str(path.relative_to(root))
        if path.stat().st_size>self.config.max_file_bytes: return [self._issue(WMQualityIssueFamily.PERFORMANCE_RISK,WMQualitySeverity.MEDIUM,rel,'File is oversized for bounded inspection and was skipped.','file_size_bound',metadata={'bytes':path.stat().st_size})]
        return self.classify_source_text(rel, path.read_text(encoding='utf-8',errors='replace'))
    def classify_tree(self, root: Path, include_globs: Optional[List[str]]=None):
        out=WMQualityIssueSet(max_issues=self.config.max_issues); patterns=include_globs or ['mnemonic_cortex/working_memory/**/*.py','tests/test_wm*.py']
        files=[]
        for pat in patterns: files += list(root.glob(pat))
        uniq=[]; seen=set()
        for p in files:
            if p.is_file() and p not in seen and '__pycache__' not in p.parts: seen.add(p); uniq.append(p)
        for p in sorted(uniq)[:self.config.max_files]: out.extend(self.classify_source_file(root,p) if p.suffix=='.py' else [])
        if len(uniq)>self.config.max_files: out.truncated=True
        return out
    def classify_pytest_output(self, text: str, file_path: str='docs/qdt_wm_maae/pytest_output.txt'):
        if not text: return [self._issue(WMQualityIssueFamily.WEAK_TEST_COVERAGE,WMQualitySeverity.MEDIUM,file_path,'Missing pytest output; cannot verify test health.','pytest_missing')]
        if re.search(r'\d+\s+(failed|error)', text.lower()): return [self._issue(WMQualityIssueFamily.WEAK_TEST_COVERAGE,WMQualitySeverity.BLOCKER,file_path,'Pytest output indicates test failures/errors.','pytest_output',text[-500:])]
        return []
    def classify_benchmark_output(self, text: str, file_path: str='docs/qdt_wm_maae/benchmark_output.txt'):
        if not text: return [self._issue(WMQualityIssueFamily.PERFORMANCE_RISK,WMQualitySeverity.LOW,file_path,'Missing benchmark output; performance smoke status is unknown.','benchmark_missing')]
        if 'pass' in text.lower() and 'false' in text.lower(): return [self._issue(WMQualityIssueFamily.PERFORMANCE_RISK,WMQualitySeverity.HIGH,file_path,'Benchmark output indicates failing smoke benchmark.','benchmark_output',text[-500:])]
        return []
    def classify_tracker_text(self, text: str, file_path: str):
        lower=text.lower(); issues=[]
        if 'partially_complete' in lower or 'pending' in lower:
            fam=WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP if 'real enhancedmnemoniccortex source' in lower or 'real source' in lower else WMQualityIssueFamily.PRODUCTION_HARDENING_DEFERRAL
            sev=WMQualitySeverity.HIGH if fam==WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP else WMQualitySeverity.MEDIUM
            issues.append(self._issue(fam,sev,file_path,'Tracker/deferred register contains incomplete or pending items.','tracker_deferred_text',lower[:800]))
        return issues
