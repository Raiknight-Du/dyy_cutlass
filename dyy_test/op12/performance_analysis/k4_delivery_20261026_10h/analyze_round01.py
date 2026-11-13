"""Offline paired audit; no CUDA imports and no frequency conversion."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.run/'PAIRED_CLOCK_RAW.json').read_text())
event=json.loads((a.run/'EVENT_QC.json').read_text())
stage=json.loads((a.run/'PAIRED_STAGE_QC.json').read_text())
rng=np.random.default_rng(20261026)
def stats(xs):
    x=np.array(xs,dtype=float);lo,hi=np.percentile(x,[5,95]);med=float(np.median(x))
    return dict(n=len(xs),median=med,mean=float(np.mean(x)),p5=float(lo),p95=float(hi),cv_pct=float(np.std(x)/np.mean(x)*100))
def interval(xs):
    x=np.array(xs);ci=np.percentile(np.median(x[rng.integers(0,len(x),(10000,len(x)))],axis=1),[2.5,97.5]).tolist()
    return dict(median=statistics.median(xs),CI95=ci,raw=xs)
checks=[]
for path in (a.run/'PAIRED_RAW').glob('*.npz'):
    x=np.load(path)['timestamps'];cb=x[4,:64,0];qb=x[9,:64,0];tb=x[9,:64,2]
    assert np.all(cb>=0) and np.all(qb>=0) and np.all(tb>=0)
    assert np.all(np.diff(cb)>=0) and np.all(np.diff(qb)>=0) and np.all(np.diff(tb)>=0)
    sm={int(x[4,64,7]),*[int(v) for v in x[4,:64,7]],*[int(v) for v in x[9,:64,7]]}
    assert len(sm)==1 and min(sm)>=0
    checks.append(dict(file=path.name,SMID=next(iter(sm)),events=193))
out={}
for scope in ['CP_EVENT','K4_EVENT']:
    rr=[r for r in rows if r['series']==scope and r['pair']=='FRAME0__FRAME7']
    summaries={}
    for variant in ['FRAME0','FRAME7']:
        vr=[r for r in rr if r['variant']==variant]
        summaries[variant]={k:stats([r[k] for r in vr]) for k in ['INIT_cycles','INIT_end_to_CG1_first']}
        summaries[variant]['INIT_BEGIN_to_CG1_FIRST_RETURN']=stats([r['first_CG1_return']-r['INIT_begin'] for r in vr])
        summaries[variant]['INIT_BEGIN_to_CG1_LAST_RETURN']=stats([r['last_CG1_return']-r['INIT_begin'] for r in vr])
    deltas={k:[] for k in ['INIT_cycles','INIT_end_to_CG1_first','INIT_BEGIN_to_CG1_FIRST_RETURN']}
    for block in sorted({r['block'] for r in rr}):
        by={v:[r for r in rr if r['block']==block and r['variant']==v] for v in ['FRAME0','FRAME7']}
        for k in deltas:
            get=(lambda r:r['first_CG1_return']-r['INIT_begin']) if k=='INIT_BEGIN_to_CG1_FIRST_RETURN' else (lambda r:r[k])
            deltas[k].append(statistics.mean(get(r) for r in by['FRAME7'])-statistics.mean(get(r) for r in by['FRAME0']))
    out[scope]=dict(distributions=summaries,paired_signed_FRAME7_minus_FRAME0_cycles={k:interval(v) for k,v in deltas.items()})
report=dict(status='OFFLINE_AUDITED',source=str(a.run),event=event,paired_INIT=stage,cycles=out,
    raw_calls_checked=len(checks),all_calls_193_events_same_CTA_same_SM=True,
    correct=json.loads((a.run/'CORRECTNESS.json').read_text()),local_absolute_bias_bound_cycles=None,
    definitions=dict(INIT='W4 _load_cp_initial_state caller clock window',startup='INIT_END to first CG1 helper return; includes waits, preparation, computation',
    total_first='same invocation INIT_BEGIN to first CG1 helper return',signed_delta='FRAME7 minus FRAME0; negative INIT means shortening'),
    warning='Screen is not formal evidence. Production marker timestamps are helper returns, not first TMA issue/completion. INIT/state-ready endpoints differ slightly. No Event-to-cycle conversion.',model_enabled=False)
(a.run/'OFFLINE_ANALYSIS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(a.run/'RAW_CALL_AUDIT.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(dict(event=event,paired_INIT=stage,cycles=out),indent=2))
