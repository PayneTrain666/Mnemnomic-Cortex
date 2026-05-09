from __future__ import annotations
from pathlib import Path
import importlib.util, inspect, json, tempfile, traceback, time, signal, sys

BASE = Path(__file__).resolve().parent
TEST_DIR = BASE / 'tests'
RESULT_JSON = BASE / 'docs/qdt_wm_maae_quality/42_wm_qd6a_test_fallback_results.json'
RESULT_TXT = BASE / 'docs/qdt_wm_maae_quality/42_wm_qd6a_pytest_output.txt'

class Timeout(Exception): pass

def alarm_handler(signum, frame):
    raise Timeout('test timeout')

def load_module(path: Path):
    name = 'wmqd6a_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def build_args(sig):
    args = []
    cleanup = []
    for pname, param in sig.parameters.items():
        if pname == 'tmp_path':
            td = tempfile.TemporaryDirectory()
            cleanup.append(td)
            args.append(Path(td.name))
        else:
            raise RuntimeError(f'Unsupported fixture/argument: {pname}')
    return args, cleanup

def run_all():
    results=[]
    start=time.time()
    for path in sorted(TEST_DIR.glob('test_*.py')):
        module_result={'file':str(path.relative_to(BASE)), 'import_ok': False, 'tests': []}
        try:
            mod=load_module(path)
            module_result['import_ok']=True
        except Exception as exc:
            module_result['import_error']=traceback.format_exc()
            results.append(module_result)
            continue
        for name in sorted(n for n in dir(mod) if n.startswith('test_')):
            obj=getattr(mod, name)
            if not callable(obj):
                continue
            row={'name':name, 'pass':False}
            cleanup=[]
            t0=time.time()
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(30)
            try:
                args, cleanup = build_args(inspect.signature(obj))
                obj(*args)
                row['pass']=True
            except Exception:
                row['error']=traceback.format_exc()
            finally:
                signal.alarm(0)
                for c in cleanup:
                    try: c.cleanup()
                    except Exception: pass
                row['duration_s']=time.time()-t0
            module_result['tests'].append(row)
        results.append(module_result)
    total=sum(len(m['tests']) for m in results)
    passed=sum(1 for m in results for t in m['tests'] if t['pass'])
    failed=total-passed + sum(1 for m in results if not m.get('import_ok'))
    out={'runner':'wm_qd6a_pytest_compatible_callable_fallback','reason':'pytest subprocess hung in this environment; test functions were imported and executed directly with tmp_path support','module_count':len(results),'test_count':total,'passed':passed,'failed':failed,'all_pass':failed==0,'duration_s':time.time()-start,'modules':results}
    RESULT_JSON.write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines=[f"WM-QD-6A fallback full test runner", f"modules={len(results)} tests={total} passed={passed} failed={failed} duration={out['duration_s']:.2f}s", '']
    for m in results:
        lines.append(f"MODULE {m['file']} import_ok={m.get('import_ok')}")
        if not m.get('import_ok'):
            lines.append(m.get('import_error',''))
        for t in m['tests']:
            lines.append(f"  {'PASS' if t['pass'] else 'FAIL'} {t['name']} {t['duration_s']:.3f}s")
            if not t['pass']:
                lines.append(t.get('error',''))
    RESULT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'test_count':total,'passed':passed,'failed':failed,'all_pass':out['all_pass'],'duration_s':out['duration_s']}, indent=2))
    return 0 if out['all_pass'] else 1

if __name__ == '__main__':
    raise SystemExit(run_all())
