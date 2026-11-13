"""Evidence-only accounting: paired Event deltas, strict slopes, SASS and NCU."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def val(s):
    try:return float(s.replace(',',''))
    except (AttributeError,ValueError):return 0.0
def regress(points):
    xs,ys=zip(*points); xb,yb=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-xb)*(y-yb) for x,y in points)/sum((x-xb)**2 for x in xs)
    intercept=yb-slope*xb
    residual=sum((y-(intercept+slope*x))**2 for x,y in points)
    ss=sum((y-yb)**2 for y in ys)
    return {'slope_us_per_repeat':slope,'intercept_us':intercept,'R2':1-residual/ss if ss else 0,
            'cycles_per_repeat_at_1845MHz':slope*1845}
def analyze_slopes(root,tag):
    groups=collections.defaultdict(list)
    for p in sorted(root.glob('runs/'+tag+'/*/slope_samples.csv')):
        for r in csv.DictReader(p.open()):
            groups[(p.parent.parent.name,int(r['trial']),r['style'],int(r['mode']),int(r['repeat']))].append(float(r['us']))
    curves=[]; windows=[]; rows=[]
    for (tag,trial,style,mode,repeat),vs in groups.items():
        median=statistics.median(vs); vs=sorted(vs)
        p5,p95=vs[int(.05*(len(vs)-1))],vs[int(.95*(len(vs)-1))]
        rows.append({'tag':tag,'trial':trial,'style':style,'mode':mode,'repeat':repeat,
                     'n':len(vs),'median_us':median,'p5_us':p5,'p95_us':p95,
                     'cv':statistics.pstdev(vs)/statistics.mean(vs),
                     'p5_p95_span_over_median':(p95-p5)/median})
    keys=sorted(set((r['trial'],r['style'],r['mode']) for r in rows))
    for trial,style,mode in keys:
        # Use latest tag when extended scans reuse a point, but keep raw prior points.
        selected={}
        for r in sorted(rows,key=lambda r:r['tag']):
            if (r['trial'],r['style'],r['mode'])==(trial,style,mode): selected[r['repeat']]=r
        seq=sorted(selected.values(),key=lambda r:r['repeat'])
        fits=[]
        for i in range(2,len(seq)):
            segment=seq[i-2:i+1]
            fits.append({'R':[r['repeat'] for r in segment],**regress([(r['repeat'],r['median_us']) for r in segment])})
        last=fits[-2:] if len(fits)>=2 else fits
        rel=(abs(last[-1]['slope_us_per_repeat']-last[-2]['slope_us_per_repeat'])/abs(last[-2]['slope_us_per_repeat'])) if len(last)==2 else None
        tail_ok=all(r['cv']<=.05 and r['p5_p95_span_over_median']<=.10 for r in seq[-4:])
        candidate={'trial':trial,'style':style,'mode':mode,'points':seq,'windows':fits,'tail_slope_relative_change':rel,
                   'within_process_strict_pass':len(seq)>=4 and rel is not None and rel<.01 and all(f['R2']>=.99 and f['slope_us_per_repeat']>0 for f in last) and tail_ok,
                   'noise_pass':tail_ok,'no_physical_HBM_bandwidth_claim':True}
        curves.append(candidate)
    gate=[]
    for style,mode in sorted(set((c['style'],c['mode']) for c in curves)):
        cs=[c for c in curves if c['style']==style and c['mode']==mode]
        sl=[c['windows'][-1]['slope_us_per_repeat'] for c in cs if c['windows']]
        spread=(max(sl)-min(sl))/min(sl) if sl and min(sl)>0 else None
        passed=len(cs)==3 and all(c['within_process_strict_pass'] for c in cs) and spread is not None and spread<.01
        median=statistics.median(sl) if passed else None
        gate.append({'style':style,'mode':mode,'status':'PASS' if passed else 'RETEST_REQUIRED',
                     'cross_process_slope_spread':spread,'stable_us_per_repeat':median,
                     'stable_grid32_repeat_cycles':median*1845 if median else None,
                     'scope':'32 CTA whole-kernel Event marginal helper service; not original FINAL exposed latency',
                     'candidate_slopes':sl})
    return {'raw_medians':rows,'curves':curves,'strict_convergence_gate':gate}

def analyze_ncu(root,tag):
    out=[]
    for path in sorted(root.glob('runs/'+tag+'/*_source.csv')):
        lines=path.read_text().splitlines()
        i=next(i for i,s in enumerate(lines) if s.startswith('"Address"'))
        source=[r for r in csv.DictReader(lines[i:]) if r.get('Address','').startswith('0x')]
        def sums(rs):
            keys=['Instructions Executed','Predicated-On Thread Instructions Executed','L2 Theoretical Sectors Global',
                  'L2 Theoretical Sectors Global Ideal','Warp Stall Sampling (All Samples)','stall_lg','stall_long_sb',
                  'stall_short_sb','stall_mio','stall_wait','stall_barrier',
                  'L1 Wavefronts Shared Excessive','L1 Wavefronts Shared','L1 Wavefronts Shared Ideal']
            return {k:sum(val(r.get(k)) for r in rs) for k in keys}
        op=lambda r: r['Source'].strip()
        groups={k:[r for r in source if re.search(r'\b'+k+r'[. ]',op(r))] for k in ('LDTM','STG','LDL','STL','UTCHMMA','UTMALDG','LDS','STS')}
        raw=path.with_name(path.name.replace('_source.csv','_raw.csv'))
        wide=list(csv.reader(raw.read_text().splitlines()))
        prefixes=('dram__','lts__','l1tex__','smsp__warps_eligible','smsp__issue_active','smsp__warp_issue_stalled',
                  'sm__pipe_tensor','launch__','derived__local_spilling','smsp__pcsamp')
        metrics={k:{'value':v,'unit':u} for k,u,v in zip(wide[0],wide[1],wide[2]) if k.startswith(prefixes)}
        lg=sorted(source,key=lambda r:val(r.get('stall_lg')),reverse=True)[:8]
        long=sorted(source,key=lambda r:val(r.get('stall_long_sb')),reverse=True)[:8]
        row={'name':path.stem,'pc_source_sha256':sha(path),'whole_metrics':metrics,'whole_samples':sums(source),
             'opcode_dynamic_account':{k:sums(rs) for k,rs in groups.items()},
             'top_lg':lg,'top_long_scoreboard':long,
             'top_mio':sorted(source,key=lambda r:val(r.get('stall_mio')),reverse=True)[:8],
             'warning':'All samples are PC sampling, not cycle percentages; theoretical sectors not physical HBM bytes'}
        if 'micro_profile' not in path.name:
            # Source confirmed tail; include dynamic-only taken branch, exclude setup and retire.
            indices=[j for j,r in enumerate(source) if re.search(r'\bLDTM[. ]',op(r))]
            store_indices=[j for j,r in enumerate(source) if re.search(r'\bSTG[. ]',op(r))]
            lo,hi=indices[-8],store_indices[-1]
            body=source[lo:hi+1]
            row['FINAL_candidate_bounds']=[source[lo]['Address'],source[hi]['Address']]
            row['FINAL_candidate_samples']=sums(body)
            row['FINAL_candidate_store_account']=sums([r for r in body if re.search(r'\bSTG[. ]',op(r))])
            row['FINAL_candidate_shared_STS']=sums([r for r in body if re.search(r'\bSTS[. ]',op(r))])
            row['FINAL_candidate_shared_LDS']=sums([r for r in body if re.search(r'\bLDS[. ]',op(r))])
            row['FINAL_bounds_status']='candidate bounds; verify static disassembly before stage attribution'
        out.append(row)
    return out

def audit(root,event_tag,slope_tag):
    pattern=re.compile(r'/\*([0-9a-f]+)\*/\s+(?:@!?[UP]+\d+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;')
    records=[]
    folders=list(sorted(root.glob('runs/'+event_tag+'/event_t1/binary/*')))+list(sorted(root.glob('runs/'+slope_tag+'/slope_t1/binary/MICRO_*')))
    for folder in folders:
        sass=next(folder.glob('*.sass'),None); cubin=next(folder.glob('*.cubin'),None)
        if sass is None or cubin is None or folder.name=='prefix_k1_k2':continue
        text=sass.read_text(); inst=[m.groups() for m in pattern.finditer(text)]
        ops=collections.Counter(i[1] for i in inst)
        resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
        records.append({'path':str(folder.relative_to(root)),'cubin_sha256':sha(cubin),'sass_sha256':sha(sass),
                        'normalized_instruction_sha256':hashlib.sha256('\n'.join(i[1]+' '+i[2] for i in inst).encode()).hexdigest(),
                        'resources':{k:int(v) for k,v in re.findall(r'\b(REG|STACK|SHARED|LOCAL):([0-9]+)',resource)},
                        'whole_static_counts':ops,'resource_text':resource,
                        'static_counts_are_not_dynamic':True})
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',default='analysis_v1')
    p.add_argument('--event-tag',default='event_v3'); p.add_argument('--slope-tag',default='slope_v3')
    p.add_argument('--ncu-tag',default='ncu_v1'); a=p.parse_args()
    out=a.root/a.out; out.mkdir(exist_ok=False)
    event=[]
    for path in sorted(a.root.glob('runs/'+a.event_tag+'/*/event_summary.json')):
        event.extend({'tag':path.parent.parent.name,'trial':int(path.parent.name.rsplit('t',1)[1]),**r} for r in json.loads(path.read_text()))
    result={'formal_tags':{'event':a.event_tag,'slope':a.slope_tag,'ncu':a.ncu_tag},
            'Event':event,'slopes':analyze_slopes(a.root,a.slope_tag),'NCU':analyze_ncu(a.root,a.ncu_tag),
            'binary_audit':audit(a.root,a.event_tag,a.slope_tag),
            'ledgers':[{ 'path':str(p.relative_to(a.root)),**json.loads(p.read_text())} for p in sorted(a.root.glob('runs/*/RUN_LEDGER.json'))],
            'model_changed':False,'parameters_automatically_approved':False}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'event_rows':len(event),'slopes':result['slopes']['strict_convergence_gate'],'ncu_captures':len(result['NCU']),
                      'binary_rows':len(result['binary_audit'])},indent=2))
if __name__=='__main__':main()
