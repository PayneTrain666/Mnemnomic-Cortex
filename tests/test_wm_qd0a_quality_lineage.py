from pathlib import Path
from mnemonic_cortex.working_memory.quality import build_lineage_index, infer_stage, sha256_file, build_quality_report

def test_infer_stage_from_path():
    assert infer_stage('docs/qdt_wm_maae/90_wm4c_acceptance_shipcheck.md')=='WM-4C'; assert infer_stage('unknown.py')=='unknown'

def test_lineage_index_hashes_files(tmp_path: Path):
    f=tmp_path/'mnemonic_cortex'/'working_memory'/'wm4c_file.py'; f.parent.mkdir(parents=True); f.write_text('x = 1\n')
    idx=build_lineage_index(tmp_path,'pack.zip',max_files=10).to_dict(); assert idx['ref_count']==1; ref=next(iter(idx['refs'].values())); assert ref['source_pack']=='pack.zip'; assert ref['source_hash']==sha256_file(f)

def test_quality_report_builds_serializable_output(tmp_path: Path):
    src=tmp_path/'mnemonic_cortex'/'working_memory'; src.mkdir(parents=True); (src/'foo.py').write_text('import torch\ndef f(x):\n    return x\n')
    d=tmp_path/'docs'/'qdt_wm_maae'; d.mkdir(parents=True); (d/'102_wm7a_pytest_output.txt').write_text('1 passed')
    out=build_quality_report(tmp_path,'pack.zip',max_files=10,max_issues=16,max_plans=16).to_dict(); assert out['report_id']=='wm-qdrop-0a'; assert 'issue_set' in out; assert out['safety_payload']['no_memory_store_mutation'] is True
