"""Derive isolated runner from a frozen archived complete-CP runner."""
from pathlib import Path

R=Path(__file__).resolve().parent
source=(R/'run_e01_reference.py').read_text()
source=source.replace("default='B0,B1,E'", "default='B0,B1,COARSE,FINE'")
source=source.replace("a=p.parse_args()", "p.add_argument('--sample-block',type=int,default=32);p.add_argument('--layouts',default='G0,G1');a=p.parse_args()")
source=source.replace("assert TARGET=='K2' and a.variants=='B0,B1,E', 'E01 scope only'", "assert TARGET=='K2' and set(a.variants.split(','))<=set(['B0','B1','COARSE','FINE'])")
source=source.replace("(12,66,12,2,2)","(12,48,2,2)")
source=source.replace("kernel,*args[:-1],tr,args[-1]", "kernel,*args[:-1],tr,cutlass.Int32(a.sample_block),args[-1]")
source=source.replace("fun(*runtime[:-1],trace,runtime[-1])", "fun(*runtime[:-1],trace,a.sample_block,runtime[-1])")
source=source.replace("'K2Clock'+v", "'K2Trace'+v").replace("'K2Clock'+v", "'K2Trace'+v")
source=source.replace("TARGET+'Clock'+v", "TARGET+'Trace'+v")
source=source.replace("('G0','G1')", "a.layouts.split(',')")
cut=source.index('for rep in range(a.warmup):')
source=source[:cut]+'''
for rep in range(a.warmup):
    for v in modes:
        for layout in a.layouts.split(','):graphs[v,layout,rep%4][0].replay()
    stream.synchronize()
rows=[];records=[]
stages=json.loads((R/'K2_CLOCK/GENERATION.json').read_text())['stages']
pairs=[(('B0',layout),(v,layout))for layout in a.layouts.split(',')for v in modes if v!='B0']
if 'COARSE'in modes and 'FINE'in modes:pairs += [(('COARSE',layout),('FINE',layout))for layout in a.layouts.split(',')]
for rep in range(a.samples):
    for pair_id,(aa,bb) in enumerate(pairs):
        ordering=[aa,bb,bb,aa] if (rep+pair_id+a.trial)%2==0 else [bb,aa,aa,bb]
        bi=rep%4
        for position,(v,layout)in enumerate(ordering):
            if (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists():raise RuntimeError('competition')
            g,start,end,ks,ke=graphs[v,layout,bi];g.replay();end.synchronize()
            invocation=f'{a.trial}:{pair_id}:{rep}:{position}'
            rows.append(dict(trial=a.trial,pair_id=pair_id,pair_A=':'.join(aa),pair_B=':'.join(bb),rep=rep,position=position,variant=v,layout=layout,buffer=bi,invocation=invocation,sample_block=a.sample_block,CP_Event_us=start.elapsed_time(end)*1000,K2_Event_us=ks.elapsed_time(ke)*1000 if layout=='G1'else''))
            if v in ('COARSE','FINE'):
                raw=trace.cpu().tolist()
                for w in range(12):
                    for sid,label in stages.items():
                        s=int(sid);b,e=raw[w][s]
                        if b[0]<0 and e[0]<0:continue
                        assert 0<=b[0]<=e[0] and b[1]==e[1],(w,s,b,e)
                        records.append(dict(trial=a.trial,invocation=invocation,variant=v,layout=layout,S=a.s,cta=0,warp=w,lane=0,iteration=a.sample_block if s>=3 else -1,stage_id=s,stage=label,smid=b[1],start_cycle=str(b[0]),end_cycle=str(e[0]),cycles=e[0]-b[0],binary_sha256=artifacts[v]['cubin_sha256']))
            for key,gi in [('output',0),('state',1)]:assert torch.equal(buffers[bi][key],gold[bi][gi])
    if rep%20==19:print(json.dumps(dict(trial=a.trial,block=a.sample_block,blocks=rep+1,rows=len(rows),records=len(records))),flush=True)
save('CLOCK_RECORDS.json',records)
with(out/'EVENT_SAMPLES.csv').open('w',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
assert torch.equal(initial,shared['initial'])
save('SUMMARY.json',dict(records=len(records),event_rows=len(rows),sample_block=a.sample_block,correctness_exact=True,MN_workspace_correctness_exact=True,model_enabled=False,local_bias_bound=None))
assert not cmd('git','-C',str(REPO),'status','--porcelain')
print('TRACE_RUN_COMPLETE',flush=True)
'''
assert "getattr(variants,TARGET+'Trace'+v)" in source
assert 'cutlass.Int32(a.sample_block)'in source
(R/'run_trace.py').write_text(source)
print('RUNNER_READY',flush=True)
