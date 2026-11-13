"""Stdlib independent-domain slope and strict convergence checks; CPU only."""
from pathlib import Path
import argparse,collections,csv,json,math,statistics
def quant(v,p):
    v=sorted(v);x=p*(len(v)-1);a=int(x);b=min(a+1,len(v)-1);return v[a]+(v[b]-v[a])*(x-a)
def fit(x,y):
    xm=statistics.mean(x);ym=statistics.mean(y)
    b=sum((a-xm)*(v-ym) for a,v in zip(x,y))/sum((a-xm)**2 for a in x)
    a=ym-b*xm;total=sum((v-ym)**2 for v in y)
    r2=1-sum((v-a-b*q)**2 for q,v in zip(x,y))/total if total else 0
    return {'slope':b,'intercept':a,'r2':r2,'N':x}
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();group=collections.defaultdict(dict);files=sorted(a.input.glob('trial*/EVENT_RAW.csv'))
    assert len(files)==3,'Formal convergence requires exactly 3 independent process files'
    for f in files:
        with f.open(newline='') as h:
            for r in csv.DictReader(h):
                assert r['valid']=='1';key=(int(r['trial']),r['config'],int(r['N']))
                group[key][int(r['sample'])]=r
    records=[]
    configs=['S_G1_LOAD','S_G128_LOAD','V_G1_LOAD','V_G128_LOAD']
    fields=['event_us','cta_cycle_median']
    for config in configs:
        grid=int(config.split('_')[1][1:]);vector=config.startswith('V');control=config.replace('LOAD','CONTROL')
        for field in fields:
            for corrected in [False,True]:
                trials=[]
                for trial in range(3):
                    ns=sorted(n for t,c,n in group if t==trial and c==config)
                    assert len(ns)>=4 and len(ns)==len(set(ns));points=[]
                    for n in ns:
                        load=group[trial,config,n];ctl=group[trial,control,n]
                        assert len(load)==30 and load.keys()==ctl.keys()
                        values=[float(load[s][field])-(float(ctl[s][field]) if corrected else 0) for s in load]
                        med=statistics.median(values);mean=statistics.mean(values)
                        points.append({'N':n,'median':med,'mean':mean,'CV':statistics.pstdev(values)/abs(mean) if mean else None,
                          'P95_P5_over_median':(quant(values,.95)-quant(values,.05))/abs(med) if med else None})
                    windows=[]
                    for sub in [points[-4:-1],points[-3:]]:
                        windows.append(fit([x['N'] for x in sub],[x['median'] for x in sub]))
                    delta=abs(windows[1]['slope']/windows[0]['slope']-1) if windows[0]['slope'] else None
                    valid=all(x['CV'] is not None and x['CV']<=.05 and x['P95_P5_over_median']<=.10 for x in points)
                    valid=valid and all(x['slope']>0 and x['r2']>=.99 for x in windows) and delta is not None and delta<.01
                    trials.append({'trial':trial,'points':points,'windows':windows,'adjacent_window_relative_change':delta,'pass':valid})
                slopes=[x['windows'][1]['slope'] for x in trials];mean=statistics.mean(slopes)
                cross=(max(slopes)-min(slopes))/abs(mean) if mean else None
                passed=all(x['pass'] for x in trials) and cross is not None and cross<.01
                numerator=65536 if field=='cta_cycle_median' else 65536*grid
                factor=1 if field=='cta_cycle_median' else 1e6
                records.append({'configuration':config,'domain':field,'control_corrected':corrected,
                 'trials':trials,'cross_process_range_over_mean':cross,'pass':passed,
                 'logical_B_per_cycle' if field=='cta_cycle_median' else 'logical_B_per_second':numerator/mean*factor if passed else None,
                 'theoretical_request_B_per_cycle' if field=='cta_cycle_median' else 'theoretical_request_B_per_second':
                    numerator*(2 if vector else 8)/mean*factor if passed else None,
                 'actual_L2_request_rate':None,'admitted_model_parameter':None})
    result={'status':'CONVERGED_CONDITIONAL_CANDIDATES' if all(x['pass'] for x in records) else 'RETEST_REQUIRED',
      'criterion':'tail adjacent3-point slope relative change<1%; process range/mean<1%; both raw/control-corrected domains; R2>=.99; all point CV<=5%, P95-P5/median<=10%',
      'primary_cycle_statistic':'per-launch median of same-CTA durations; no cross-SM timestamp subtraction',
      'records':records,'model_enabled':False,'full_prediction':None}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'candidates':len(records),'model_enabled':False}))
if __name__=='__main__':main()
