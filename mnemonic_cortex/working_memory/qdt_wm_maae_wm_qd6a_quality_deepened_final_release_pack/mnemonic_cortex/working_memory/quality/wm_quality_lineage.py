from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib, re
from .wm_quality_issue_schema import WMQualityLineageRef

_STAGE_PATTERNS=[(re.compile(k,re.I),v) for k,v in [('wmr0a|WM-R0A','WM-R0A'),('wm0b|WM-0B','WM-0B'),('wm1a|WM-1A','WM-1A'),('wm1b|WM-1B','WM-1B'),('wm1c|WM-1C','WM-1C'),('wm1d|WM-1D','WM-1D'),('wm1e|WM-1E','WM-1E'),('wm2a|WM-2A','WM-2A'),('wm2b|WM-2B','WM-2B'),('wm2c|WM-2C','WM-2C'),('wm3a|WM-3A','WM-3A'),('wm3b|WM-3B','WM-3B'),('wm4a|WM-4A','WM-4A'),('wm4b|WM-4B','WM-4B'),('wm4c|WM-4C','WM-4C'),('wm5a|WM-5A','WM-5A'),('wm6a|WM-6A','WM-6A'),('wm7a|WM-7A','WM-7A'),('wm_qd0a|WM-QD-0A','WM-QD-0A')]]

def infer_stage(path_or_name: str) -> str:
    for p,s in _STAGE_PATTERNS:
        if p.search(path_or_name): return s
    return 'unknown'

def sha256_file(path: Path, max_bytes: int=2_000_000) -> str:
    h=hashlib.sha256(); remaining=max_bytes
    with path.open('rb') as f:
        while remaining>0:
            chunk=f.read(min(65536,remaining))
            if not chunk: break
            h.update(chunk); remaining-=len(chunk)
    return h.hexdigest()

@dataclass
class WMQualityLineageIndex:
    source_pack: str; refs: Dict[str, WMQualityLineageRef]=field(default_factory=dict); max_files: int=512; truncated: bool=False
    def add_file(self, root: Path, path: Path, source_pack: Optional[str]=None):
        if len(self.refs)>=self.max_files: self.truncated=True; return
        rel=str(path.relative_to(root)); self.refs[rel]=WMQualityLineageRef(source_pack or self.source_pack, infer_stage(rel), rel, sha256_file(path), rel if rel.startswith('tests/') else None, rel if rel.startswith('docs/') else None, release_manifest_entry=rel if rel.startswith('release/') else None)
    def get(self, file_path: str) -> WMQualityLineageRef:
        return self.refs.get(file_path, WMQualityLineageRef(self.source_pack, infer_stage(file_path), file_path))
    def to_dict(self): return {'source_pack':self.source_pack,'ref_count':len(self.refs),'max_files':self.max_files,'truncated':self.truncated,'refs':{k:v.to_dict() for k,v in self.refs.items()}}

def build_lineage_index(root: Path, source_pack: str, max_files: int=512) -> WMQualityLineageIndex:
    idx=WMQualityLineageIndex(source_pack, max_files=max_files)
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.md','.txt','.json'}:
            idx.add_file(root,path)
    return idx
