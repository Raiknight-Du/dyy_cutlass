"""Reclassify raw CP evidence; do not mix Event and profiler clock domains."""
from pathlib import Path
import csv, hashlib, json, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
B = R.parent
ARCHIVE = B / '_superseded' / 'cp4_frequency_20261023'
OLDNAME = 'cp4_cold_l2_lock1845_20261023'
SOURCE = ARCHIVE / OLDNAME if (ARCHIVE / OLDNAME).exists() else B / OLDNAME
SHAPES = (8192, 16384, 32768)
KERNELS = ('K1', 'K2', 'K3', 'K4')
COLORS = dict(zip(KERNELS, ('#2c78aa', '#229582', '#d28c28', '#8763b2')))
evidence = {}
figure_rows = []

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def register(p):
    key = p.relative_to(B).as_posix()
    evidence[key] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return key

def load(p):
    register(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def rows(p):
    register(p)
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def csvout(p, rr):
    assert rr
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)

def location(clock, s):
    return SOURCE if clock == 'auto' or s == 32768 else R

def runpath(clock, s, mode, trial):
    return location(clock, s) / 'runs' / f'S{s}_cold_{clock}_{mode}_t{trial}'

def source_identity(clock, s, mode, trial):
    return load(runpath(clock, s, mode, trial) / 'IDENTITY.json')

def distribution(v):
    v = sorted(v); mean = st.mean(v)
    return {'median': st.median(v), 'mean': mean,
            'p5': v[int((len(v)-1)*.05)], 'p95': v[int((len(v)-1)*.95)],
            'cv_pct': 100 * st.pstdev(v)/mean, 'n': len(v)}

def category(clock, name, label):
    out = R / name
    out.mkdir(exist_ok=True)
    event_process = []; kernel_process = []; nvml_process = []; summaries = []
    selected_calls = []
    for s in SHAPES:
        cp_medians = []; kernel_medians = {k: [] for k in KERNELS}
        freq_medians = {k: [] for k in KERNELS}; profiled_cp = []
        for trial in (1, 2, 3):
            for mode in ('native', 'metrics'):
                ident = source_identity(clock, s, mode, trial)
                assert ident['S'] == s and ident['cache'] == 'cold'
                assert ident['clock_request'] == clock and ident['calls'] == 100
                assert ident['batches'] == 3 and ident['eviction_bytes'] == 530579456
                valid = load(runpath(clock, s, mode, trial) / 'CORRECTNESS.json')
                assert all(x['finite'] for x in valid)
                data = load(runpath(clock, s, mode, trial) / 'EVENT_SAMPLES.json')
                samples = [x for batch in data for x in batch['CP_Event_us']]
                assert len(samples) == 300
                ds = distribution(samples)
                event_process.append({'S': s, 'C': s//8, 'clock': clock,
                                      'trial': trial, 'mode': mode, **ds,
                                      'source': (runpath(clock,s,mode,trial)/'EVENT_SAMPLES.json').relative_to(B).as_posix()})
                (cp_medians if mode == 'native' else profiled_cp).append(ds['median'])
                span = load(runpath(clock, s, mode, trial)/'MEASUREMENT_WALL_RANGE.json')
                raw = load(location(clock,s)/f'formal_S{s}_cold_{clock}_{mode}_t{trial}_NVML.json')
                matched = [x for x in raw if 'wall_ns' in x and span['begin_wall_ns'] <= x['wall_ns'] <= span['end_wall_ns']]
                assert matched
                nvml_process.append({'S': s, 'clock': clock, 'mode': mode, 'trial': trial,
                    'sample_count': len(matched), 'SM_MHz_min': min(x['SM_MHz'] for x in matched),
                    'SM_MHz_median': st.median(x['SM_MHz'] for x in matched),
                    'SM_MHz_max': max(x['SM_MHz'] for x in matched),
                    'SW_power_cap_sample_fraction': st.mean(bool(x['throttle'] & 4) for x in matched),
                    'scope': '20ms samples in full replay window, includes eviction and inter-call gaps; not per-kernel'})
            folder = location(clock,s)/f'formal_S{s}_cold_{clock}_metrics_t{trial}'
            kr = rows(folder/'KERNEL_INTERVALS.csv')
            assert len(kr) == 1200
            for k in KERNELS:
                sel = [x for x in kr if x['kernel'] == k]
                hw = distribution([float(x['HW_us']) for x in sel])
                fq = distribution([float(x['GPC_clean_MHz']) for x in sel if x['GPC_clean_MHz']])
                kernel_medians[k].append(hw['median']); freq_medians[k].append(fq['median'])
                kernel_process.append({'S':s, 'C':s//8, 'clock':clock, 'trial':trial, 'kernel':k,
                    'NSYS_HW_median_us':hw['median'], 'GPC_median_MHz':fq['median'],
                    'GPC_coverage_median': st.median(float(x['GPC_coverage']) for x in sel),
                    'scope':'profiled original CP kernel; not independent Event'})
            if trial == 2:
                sel = [x for x in kr if x['batch'] == '1' and x['call'] == '50']
                assert [x['kernel'] for x in sel] == list(KERNELS)
                selected_calls.append((s, folder, sel))
        summaries.append({'S':s, 'C':s//8, 'P':8, 'clock':clock, 'cache':'cold_before_K1_only',
            'independent_CP_Event_median3_us': st.median(cp_medians),
            'native_process_medians_us':cp_medians,
            'profiled_CP_Event_median3_us': st.median(profiled_cp),
            'kernel_profiled': {k:{'NSYS_HW_median3_us':st.median(kernel_medians[k]),
                                   'GPC_median3_MHz':st.median(freq_medians[k])} for k in KERNELS},
            'model_enabled':False})
    csvout(out/'EVENT_PROCESS_STATISTICS.csv',event_process)
    csvout(out/'KERNEL_PROCESS_STATISTICS.csv',kernel_process)
    csvout(out/'NVML_WINDOW_STATISTICS.csv',nvml_process)
    csvout(out/'CP_EVENT_SUMMARY.csv',[{k:v for k,v in x.items() if k not in ('kernel_profiled','native_process_medians_us')} for x in summaries])
    csvout(out/'KERNEL_HW_GPC_SUMMARY.csv',[
        {'S':x['S'],'C':x['C'],'clock':clock,'kernel':k,**x['kernel_profiled'][k]}
        for x in summaries for k in KERNELS])
    save(out/'RESULTS.json',{'category':label,'definition':'cold K1, complete original CP K1-K4, P8',
        'Event_scope':'independent no-profiler CP only; excludes eviction',
        'GPC_scope':'separate NSYS process, complete in-kernel sampling windows; not exact per-SM cycle integral',
        'aggregate':'median of three process medians; frequency and timing never multiplied',
        'results':summaries})

    fig,axs=plt.subplots(3,1,figsize=(13,9),layout='constrained')
    plot_samples=[]
    for ax,(s,folder,ks) in zip(axs,selected_calls):
        lo=int(ks[0]['start_ns']); hi=int(ks[-1]['end_ns'])
        fs=[x for x in rows(folder/'GPC_FREQUENCY_RAW.csv') if lo-10000<=int(x['timestamp'])<=hi+10000]
        assert fs
        ax.plot([(int(x['timestamp'])-lo)/1000 for x in fs], [float(x['value'])/1e6 for x in fs],'.-',color='#20374c',ms=3,lw=1)
        ax.axhline(1845,ls='--',color='#bd4141',lw=.9,label='1845 MHz reference'+(' / lock request' if clock!='auto' else ' (not a lock)'))
        for k in ks:
            start,end=(int(k['start_ns'])-lo)/1000,(int(k['end_ns'])-lo)/1000
            ax.axvspan(start,end,alpha=.15,color=COLORS[k['kernel']])
            ax.text((start+end)/2,.9,k['kernel'],ha='center',transform=ax.get_xaxis_transform(),color=COLORS[k['kernel']],weight='bold')
            figure_rows.append({'category':name,'S':s,'clock':clock,'trial':2,'batch':1,'call':50,
                'range':k['range'],'kernel':k['kernel'],'start_ns':k['start_ns'],'end_ns':k['end_ns'],
                'source':(folder/'GPC_FREQUENCY_RAW.csv').relative_to(B).as_posix()})
        for x in fs: plot_samples.append({'S':s,'relative_us':(int(x['timestamp'])-lo)/1000,'GPC_MHz':float(x['value'])/1e6})
        ax.set(xlim=(0,(hi-lo)/1000),ylim=(1550,2020),ylabel='GPC metric (MHz)',
               title=f'S{s} / C{s//8} / P8 | cold K1 | {clock} | trial 2, batch 1, call 50')
        ax.grid(alpha=.2)
    axs[0].legend(loc='lower left',fontsize=8)
    axs[-1].set_xlabel('Time since K1 hardware start (us); eviction finishes before x=0')
    fig.suptitle(label+' | sampled GPC frequency\nSeparate profiling process: do not use this curve to convert independent Event latency into cycles',fontsize=12)
    for ext in ('png','svg'): fig.savefig(out/f'GPC_FREQUENCY_TIMELINE.{ext}',dpi=170)
    plt.close(fig); csvout(out/'GPC_PLOT_SAMPLES.csv',plot_samples)

    fig,axs=plt.subplots(3,1,figsize=(13,9),layout='constrained')
    native_plots=[]
    for ax,s in zip(axs,SHAPES):
        rp=runpath(clock,s,'native',2)
        span=load(rp/'MEASUREMENT_WALL_RANGE.json'); ev=load(rp/'EVENT_SAMPLES.json')
        rr=load(location(clock,s)/f'formal_S{s}_cold_{clock}_native_t2_NVML.json')
        lo,hi=span['begin_wall_ns'],span['end_wall_ns']
        measured=[x for x in rr if 'wall_ns' in x and lo<=x['wall_ns']<=hi]
        xx=[(x['wall_ns']-lo)/1e6 for x in measured]; yy=[x['SM_MHz'] for x in measured]
        ax.plot(xx,yy,'o-',color='#236e98',ms=5,lw=1)
        for b in ev:
            a,z=(b['host_begin_wall_ns']-lo)/1e6,(b['host_end_wall_ns']-lo)/1e6
            ax.axvspan(a,z,alpha=.08,color=COLORS['K1']);ax.text((a+z)/2,.88,f"batch {b['batch']}",ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
        for x in measured:
            native_plots.append({'S':s,'trial':2,'relative_ms':(x['wall_ns']-lo)/1e6,
                'wall_ns':x['wall_ns'],'SM_MHz':x['SM_MHz'],'throttle_mask':x['throttle']})
        event=next(x for x in summaries if x['S']==s)['independent_CP_Event_median3_us']
        ax.set(xlim=(0,(hi-lo)/1e6),ylim=(1750,2010),ylabel='NVML SM MHz',
            title=f'S{s} | native process, trial 2 | {len(measured)} samples | CP Event median3 {event:.3f} us')
        ax.axhline(1845,ls='--',color='#bd4141',lw=.9);ax.grid(alpha=.2)
    axs[-1].set_xlabel('Time since measured replay window begins (ms); includes eviction and host gaps')
    fig.suptitle(label+' | no-profiler NVML frequency\n20-ms telemetry: these sparse samples do not resolve individual kernels',fontsize=12)
    for ext in ('png','svg'): fig.savefig(out/f'NATIVE_NVML_FREQUENCY_TIMELINE.{ext}',dpi=170)
    plt.close(fig);csvout(out/'NATIVE_NVML_PLOT_SAMPLES.csv',native_plots)
    return summaries

if __name__=='__main__':
    for source in (SOURCE, R):
        for name in ('FORMAL_CLOCK_BEFORE.json','FORMAL_CLOCK_CHANGES.json','FORMAL_CLOCK_RESTORED.json','FORMAL_STATUS.json'):
            load(source/name)
    load(SOURCE/'CACHE_VALIDATION.json')
    performance=category('auto','01_performance_cold_auto','Performance: cold K1, automatic clocks')
    theory=category('locked1845','02_theory_cold_locked1845','Theory validation: cold K1, lock request 1845 MHz')
    csvout(R/'FIGURE_CALL_IDENTITIES.csv',figure_rows)
    comparison=[]
    for a,b in zip(performance,theory):
        comparison.append({'S':a['S'],'C':a['C'],
            'performance_cold_auto_CP_Event_us':a['independent_CP_Event_median3_us'],
            'theory_cold_locked1845_CP_Event_us':b['independent_CP_Event_median3_us'],
            'locked_latency_change_percent':100*(b['independent_CP_Event_median3_us']/a['independent_CP_Event_median3_us']-1)})
    csvout(R/'TWO_CATEGORY_COMPARISON.csv',comparison)
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    x=range(3); a=[z['performance_cold_auto_CP_Event_us'] for z in comparison];b=[z['theory_cold_locked1845_CP_Event_us'] for z in comparison]
    ax.bar([i-.18 for i in x],a,.34,label='Performance: cold / auto',color='#287fa4')
    ax.bar([i+.18 for i in x],b,.34,label='Theory validation: cold / locked1845',color='#d39132')
    for i in x:
        ax.text(i-.18,a[i]+4,f'{a[i]:.3f}',ha='center',fontsize=9);ax.text(i+.18,b[i]+4,f'{b[i]:.3f}',ha='center',fontsize=9)
    ax.set(xticks=list(x),xticklabels=[f'S{s}' for s in SHAPES],ylabel='Complete CP CUDA Event (us)',ylim=(0,max(a+b)*1.16),title='Same cold-K1 boundary; two clock policies\nIndependent Event only; eviction excluded; median of three process medians')
    ax.legend();ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg'):fig.savefig(R/f'CP_EVENT_TWO_CATEGORIES.{ext}',dpi=170)
    plt.close(fig)
    save(R/'EVIDENCE_INDEX.json',evidence)
    save(R/'CLASSIFICATION.json',{'user_policy':'Performance=cold/auto; theory=cold/locked1845; full-model test cancelled by user',
        'shapes':list(SHAPES),'categories':['01_performance_cold_auto','02_theory_cold_locked1845'],
        'no_eviction_data':'historical only; not used for current conclusions',
        'old_evidence_archive':'_superseded/cp4_frequency_20261023',
        'new_gpu_configurations':2,'new_formal_processes':12,'model_enabled':False})
    print(json.dumps(comparison,indent=2))
