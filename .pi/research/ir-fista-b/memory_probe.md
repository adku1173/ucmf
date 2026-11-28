# Exact post-Gate-2 isolated-process RSS probe

This successful parent numerical measurement produced `memory_isolated.jsonl`. Preserve it as a reproducibility artifact, not as package source. It uses the reviewed benchmark harness rather than another solver implementation. Run from repository root after Gate 2. Re-running overwrites the named result artifact. Cold-start OS RSS, not pure tensor allocation; scope includes final external oracle. See `benchmark_notes.md` for interpretation and the earlier parser error.

```bash
uv run python - <<'PY'
import json, subprocess, sys
from pathlib import Path
worker = r'''
import gc, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
spec=importlib.util.spec_from_file_location('ir_bench','examples/benchmark_ir_fista_b.py')
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
family,solver=sys.argv[1:]
a=SimpleNamespace(seed=7957,dtype='float64',device='cpu',large=True,m=512,n=8192,max_mib=512,mics=16,grid=32,support=3,signed=False,lam_fraction=.1,cap=10.,inner_steps=100,outer_steps=200,steps=10000,check_every=10,mode='benchmark')
torch.set_num_threads(1)
def rss(key):
    return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith(key+':')))
with torch.no_grad():
    op,y,ref_op,y_ref,lam,scale=h.make_problem(a,family)
    gc.collect()
    before=rss('VmRSS')
    reset=False
    try:
        Path('/proc/self/clear_refs').write_text('5\n'); reset=True
    except OSError:
        pass
    q=h.run_solver(solver,op,y,ref_op,y_ref,lam,1e-6*scale,a)
    peak=rss('VmHWM')
    out=dict(family=family,solver=solver,shape=op.shape,rss_before_kib=before,peak_rss_kib=peak,reset_peak=reset,incremental_peak_kib=peak-before if reset else None,success_oracle=q['success_oracle'],wall_seconds=q['wall_seconds'],natural=q['quality']['natural'].tolist(),raw=q['quality']['raw'].tolist(),scope='fresh process; after instance generation; public solve + final external oracle',caveat='OS RSS high-water, includes lazy pages and allocator effects; not tensor bytes or a universal memory bound')
    print(json.dumps(out))
'''
records=[]
for family in ['dense','cmf']:
    for solver in ['IRFISTAB','FISTA','ISTA']:
        p=subprocess.run([sys.executable,'-c',worker,family,solver],capture_output=True,text=True,timeout=40)
        if p.returncode:
            print(p.stderr); raise RuntimeError((family,solver,p.returncode))
        result=json.loads(p.stdout); records.append(result)
        print(json.dumps(result))
path=Path('.pi/research/ir-fista-b/memory_isolated.jsonl')
path.write_text(''.join(json.dumps(q)+'\n' for q in records))
PY
```
