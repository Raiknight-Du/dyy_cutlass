from pathlib import Path
import argparse,bisect,csv,json,sqlite3,statistics as st,subprocess
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('stem');p.add_argument('run');a=p.parse_args();report=R/a.stem;run=R/'runs'/a.run;db=report.with_suffix('.sqlite')
def load(p):return json.loads(p.read_text())
if not db.exists():
    with report.with_suffix('.export.log').open('w') as f:subprocess.run(['nsys','export','--type=sqlite','--output',str(db),str(report.with_suffix('.nsys-rep'))],check=True,stdout=f,stderr=f)
c=sqlite3.connect(db);c.row_factory=sqlite3.Row;names={x['id']:x['value'] for x in c.execute('select * from StringIds')};symbols=[('K1','CPDeltaRuleTPrecomputeSm100'),('K2','CPDeltaRuleMNPrecomputeUtcmma1Sm100'),('K3','CPDeltaRuleFixupUtcmmaSm100'),('K4','CPDeltaRulePrefillTcgen05Sm100')]
allks=[dict(x) for x in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
for k in allks:k['role']=next((r for r,s in symbols if s in names[k['demangledName']]),'evict')
ranges={x['text']:dict(x) for x in c.execute('select * from NVTX_EVENTS') if x['text'] and x['text'].startswith('CP4C_')}
meta=[dict(x) for x in c.execute('select * from TARGET_INFO_GPU_METRICS') if 'GPC Clock Frequency' in x['metricName']];assert len(meta)==1
freq=[dict(x) for x in c.execute('select timestamp,rawTimestamp,value from GPU_METRICS where metricId=? and typeId=? order by timestamp',(meta[0]['metricId'],meta[0]['typeId']))];times=[x['timestamp'] for x in freq];dt=st.median(y-x for x,y in zip(times,times[1:]));assert 1e8<st.median(x['value'] for x in freq)<3e9
def fdata(start,end):
    ft=0.;ct=0;vs=[];gaps=0
    for i in range(max(1,bisect.bisect_right(times,start)),bisect.bisect_right(times,end)):
        lo,hi=times[i-1],times[i]
        if not start<=lo<hi<=end:continue
        if hi-lo>dt*1.5:gaps+=1;continue
        ft+=freq[i]['value']*(hi-lo);ct+=hi-lo;vs.append(freq[i]['value']/1e6)
    return {'GPC_clean_MHz':ft/ct/1e6 if ct else None,'GPC_coverage':ct/(end-start),'GPC_clean_intervals':len(vs),'GPC_clean_min_MHz':min(vs) if vs else None,'GPC_clean_max_MHz':max(vs) if vs else None,'GPC_gap_intervals':gaps,'frequency_weight_ns':ct,'frequency_fdt_MHz_ns':ft/1e6}
ident=load(run/'IDENTITY.json');ev=load(run/'EVENT_SAMPLES.json');rows=[];calls=[];audits=[]
for batch in ev:
    nr=ranges[batch['range']];kk=[k for k in allks if nr['start']<=k['start'] and k['end']<=nr['end']];ks=[k for k in kk if k['role']!='evict'];assert len(ks)==4*ident['calls']
    assert [k['role'] for k in ks]==['K1','K2','K3','K4']*ident['calls']
    extra=[k for k in kk if k['role']=='evict'];assert len(extra)==ident['calls']*(ident['cache']=='cold')
    overlaps=[x['end']-y['start'] for x,y in zip(ks,ks[1:]) if x['end']>y['start']]
    audits.append({'range':batch['range'],'kernel_count':len(kk),'CP_calls':ident['calls'],'eviction_kernels':len(extra),'reported_overlap_count':len(overlaps),'reported_overlap_max_ns':max(overlaps,default=0),'note':'endpoints verbatim; graph order preserved, no clipping'})
    for call in range(ident['calls']):
        four=ks[4*call:4*call+4]
        if extra:assert extra[call]['end']<=four[0]['start']+1000
        base={'S':ident['S'],'C':ident['C'],'cache':ident['cache'],'clock_request':ident['clock_request'],'trial':ident['trial'],'range':batch['range'],'batch':batch['batch'],'call':call,'buffer':call%4}
        cr={**base,'CP_Event_us':batch['CP_Event_us'][call],'CP_HW_span_us':(four[-1]['end']-four[0]['start'])/1000,'eviction_HW_us':(extra[call]['end']-extra[call]['start'])/1000 if extra else None}
        for k in four:
            r={**base,'kernel':k['role'],'start_ns':k['start'],'end_ns':k['end'],'HW_us':(k['end']-k['start'])/1000,'gridId':k['gridId'],'graphNodeId':k['graphNodeId'],'grid':k['gridX'],'threads':k['blockX'],'registers':k['registersPerThread'],'dynamic_smem':k['dynamicSharedMemory']};r.update(fdata(k['start'],k['end']));rows.append(r)
        calls.append(cr)
def stats(x):
    y=sorted(v for v in x if v is not None)
    if not y:return None
    mean=st.mean(y);return {'n':len(y),'median':st.median(y),'mean':mean,'p5':y[int((len(y)-1)*.05)],'p95':y[int((len(y)-1)*.95)],'min':y[0],'max':y[-1],'cv_percent':100*st.pstdev(y)/mean if mean else None}
summary={'identity':ident,'freq_sample_interval_ns':dt,'frequency_metric':meta[0],'sample_count':len(freq),'kernel_count':len(rows),'audits':audits,'kernels':{}}
for key in ('K1','K2','K3','K4'):
    sel=[x for x in rows if x['kernel']==key];summary['kernels'][key]={field:stats([x[field] for x in sel]) for field in ('HW_us','GPC_clean_MHz','GPC_coverage','GPC_clean_intervals')};summary['kernels'][key]['time_weighted_GPC_MHz']=sum(x['frequency_fdt_MHz_ns'] for x in sel)/sum(x['frequency_weight_ns'] for x in sel);summary['kernels'][key]['no_clean_samples']=sum(x['GPC_clean_intervals']==0 for x in sel)
out=report.with_suffix('');out.mkdir(exist_ok=False)
def csvout(name,rr):
    with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
csvout('KERNEL_INTERVALS.csv',rows);csvout('CP_CALLS.csv',calls);csvout('GPC_FREQUENCY_RAW.csv',freq);(out/'ANALYSIS.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'report':a.stem,'kernel_count':len(rows),'sample_dt_ns':dt,'kernels':{k:{f:v[f]['median'] for f in ('HW_us','GPC_clean_MHz','GPC_coverage')} for k,v in summary['kernels'].items()}},indent=2))
