"""Per-invocation original-return envelopes. No synthetic cross-call timeline."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--shape',type=int,choices=[8192,16384,32768],required=True);p.add_argument('--variant',choices=['CLOCK_RETURNS','PBASE'],default='CLOCK_RETURNS');a=p.parse_args()
L=a.shape//8//64
middle={16:(4,11),32:(8,23),64:(8,55)}[L]
metadata={r['invocation']:r for r in json.loads((a.run/'PAIRED_CLOCK_RAW.json').read_text())}
rows=[];intervals=[]
for invocation,rec in metadata.items():
    if rec['variant']!=a.variant:continue
    path=a.run/'PAIRED_RAW'/(invocation+'.npz');arr=np.load(path)['timestamps']
    cg=arr[4,:L,0];q=arr[9,:L,0];t=arr[9,:L,2]
    ib,ie=int(arr[4,64,0]),int(arr[4,64,1])
    assert 0<=ib<=ie and np.all(cg>=ie) and np.all(q>=0) and np.all(t>=0)
    assert np.all(np.diff(cg)>=0) and np.all(np.diff(q)>=0) and np.all(np.diff(t)>=0)
    assert np.all(arr[4,L:64,0]==-1) and np.all(arr[9,L:64,0]==-1) and np.all(arr[9,L:64,2]==-1)
    sm={int(arr[4,64,7]),*[int(x) for x in arr[4,:L,7]],*[int(x) for x in arr[9,:L,7]]}
    assert len(sm)==1 and min(sm)>=0
    # Index k names the return at k; its inter-return interval begins at k-1.
    delta=np.concatenate([[int(cg[0])-ie],np.diff(cg)])
    mid=float(np.mean(delta[middle[0]:middle[1]+1]));tail=int(delta[L-1])
    row=dict(invocation=invocation,process=rec['process'],series=rec['series'],block=rec['block'],member=rec['member'],buffer=rec['buffer'],
        S=a.shape,C=a.shape//8,L=L,CTA=16,SMID=next(iter(sm)),events=1+3*L,cubin=rec['cubin'],
        INIT_cycles=ie-ib,INIT_END_to_first_CG1_RETURN=int(cg[0])-ie,
        INIT_BEGIN_to_last_CG1_RETURN=int(cg[-1])-ib,first_to_last_CG1_RETURN=int(cg[-1])-int(cg[0]),
        middle_first_k=middle[0],middle_last_k=middle[1],middle_mean_inter_return_cycles=mid,
        tail_inter_return_cycles=tail,tail_minus_middle_cycles=tail-mid,tail_over_middle=tail/mid,
        first_QKV_helper_RETURN_minus_INIT_END=int(q[0])-ie,first_T_helper_RETURN_minus_INIT_END=int(t[0])-ie)
    rows.append(row)
    intervals.append(dict(invocation=invocation,CG1_return_cycles=[int(x) for x in cg],interval_k0_from_INIT_END_then_previous_return=[int(x) for x in delta],
        QKV_helper_return_cycles=[int(x) for x in q],T_helper_return_cycles=[int(x) for x in t],INIT_begin=ib,INIT_end=ie))
assert rows
with (a.run/'ENDPOINT_PER_CALL.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def stats(values):
    x=np.array(values,dtype=float)
    return dict(n=len(values),median=float(np.median(x)),mean=float(np.mean(x)),p5=float(np.percentile(x,5)),p95=float(np.percentile(x,95)))
summary={}
keys=['INIT_cycles','INIT_END_to_first_CG1_RETURN','INIT_BEGIN_to_last_CG1_RETURN','first_to_last_CG1_RETURN','middle_mean_inter_return_cycles','tail_inter_return_cycles','tail_minus_middle_cycles','tail_over_middle']
for scope in ['CP_EVENT','K4_EVENT']:
    selected=[r for r in rows if r['series']==scope]
    summary[scope]={k:stats([r[k] for r in selected]) for k in keys}
report=dict(S=a.shape,C=a.shape//8,L=L,CTA=16,variant=a.variant,middle_interval_return_indices=middle,valid_calls=len(rows),
    expected_events=1+3*L,all_same_SM_per_call=True,summary=summary,
    interpretation='Within-process descriptive endpoint distributions. Helper returns are neither first TMA issue nor physical completion. Middle selection does not redefine the formal model boundary. No fitted intercept is a theoretical node.',local_absolute_bias_bound_cycles=None,model_enabled=False)
(a.run/'ENDPOINT_SUMMARY.json').write_text(json.dumps(report,indent=2))
(a.run/'CG1_ALL_RETURN_INTERVALS.json').write_text(json.dumps(intervals,indent=2))
print(json.dumps(report,indent=2))
