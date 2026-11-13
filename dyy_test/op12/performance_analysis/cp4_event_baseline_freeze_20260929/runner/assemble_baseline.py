#!/usr/bin/env python3
"""Validate real-chain evidence; exclusive-create one frozen baseline, no fitting."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='e77a4a0d276367895c3b50a642fd8f326c03fb72'
ID='OP12_CP4_EVENT_FP16_B1H16_P8_20260929_V1'

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def near(a,b): assert abs(a-b)<1e-8,(a,b)

records=read(ROOT/'formal_process_records.json')
assert len(records)==9 and all(r['exit_code']==0 for r in records)
allruns=[]
shapes=[]
check=[]
firstenv=None
for s in [8192,16384,32768]:
    runs=[]
    for t in [1,2,3]:
        rel=f'formal/S{s}_trial{t}'
        p=ROOT/rel
        env,su,inp,gv,va=[read(p/n) for n in ['environment.json','summary.json','inputs_manifest.json','graph_validation.json','validation.json']]
        assert su['status']=='PASS' and all(su['admission'].values()),(rel,su['admission'])
        assert env['S']==s and env['pin']==PIN and env['trial']==t
        assert env['warmup']==20 and env['samples']==100 and env['calls_per_graph']==400
        assert all(all(x[k] for k in ['O_exact','state_exact','O_finite','state_finite']) for x in va['cp_graph_exact'])
        assert va['initial_unchanged'] and all(x['allclose'] for x in va['non_cp_comparison'].values())
        if firstenv is None:firstenv=env
        for key in ['runner_sha256','source_sha256','torch','cuda','nvidia-cutlass-dsl','gpu_name','sm_count']:
            assert env[key]==firstenv[key],(rel,key)
        with (p/'prefix_event_samples.csv').open(newline='') as f:raw=list(csv.DictReader(f))
        assert len(raw)==500
        for k in range(5):
            selected=[r for r in raw if int(r['probe'])==k]
            assert len(selected)==100 and sorted(int(r['sample']) for r in selected)==list(range(100))
            assert [sum(int(r['buffer'])==b for r in selected) for b in range(4)]==[25]*4
            xs=[float(r['prefix_interval_us']) for r in selected]
            near(statistics.mean(xs),su['probes'][f'P{k}']['mean_us'])
        for k in range(1,5):near(su['stage_estimates'][f'K{k}']['estimate_us'],
            su['probes'][f'P{k}']['mean_us']-su['probes'][f'P{k-1}']['mean_us'])
        for g in list(gv['probes'].values())+[gv['native']]:
            assert g['kernel_nodes']==1600 and g['four_kernel_sequence_repetitions']==400
            assert g['all_nodes_single_serial_chain']
        assert all(g['event_nodes']==202 for g in gv['probes'].values()) and gv['native']['event_nodes']==2
        assert inp['content_signature_sha256']==su['input_signature']
        run={'S':s,'C':env['C'],'trial':t,'path':rel,'summary':su,
             'input_signature':su['input_signature'],'validation':va,
             'environment':env,'native_batch_samples':read(p/'native_batch_samples.json')}
        runs.append(run);allruns.append(run)
        check.append({'path':rel,'status':'PASS','raw_prefix_observations':500,'graph_kernels_each':1600})
    assert len({r['input_signature'] for r in runs})==1
    stage={k:statistics.median(r['summary']['stage_estimates'][k]['estimate_us'] for r in runs) for k in ['K1','K2','K3','K4']}
    sig=runs[0]['input_signature']
    input_root=ROOT/'frozen_inputs'/f'S{s}'
    assert read(input_root/'inputs_manifest.json')['content_signature_sha256']==sig
    store=read(input_root/'storage_hash.json')
    assert sha(input_root/'inputs.pt')==store['inputs.pt']
    # The native graph has no internal timing nodes; do not scale the stages to it.
    native=statistics.median(r['summary']['native_batch_per_call']['median_us'] for r in runs)
    shapes.append({'S':s,'C':runs[0]['C'],'P':8,'grid':{'K1':s//64*16,'K2':128,'K3':32,'K4':128},
        'stage_us':stage,'stage_sum_us':sum(stage.values()),
        'native_chain_us':native,'stage_sum_vs_native_pct':(sum(stage.values())/native-1)*100,
        'chain_estimate_median_us':statistics.median(r['summary']['chain_estimate_us'] for r in runs),
        'stage_process_minmax_us':{k:[min(r['summary']['stage_estimates'][k]['estimate_us'] for r in runs),
            max(r['summary']['stage_estimates'][k]['estimate_us'] for r in runs)] for k in stage},
        'max_probe_vs_native_abs_pct':max(r['summary']['marker_check_pct_max'] for r in runs),
        'max_nonempty_probe_cv_pct':max(r['summary']['probes'][f'P{p}']['cv_pct'] for r in runs for p in range(1,5)),
        'input_signature_sha256':sig,'frozen_input_tensor_path':str(input_root/'inputs.pt'),
        'frozen_input_storage_sha256':store['inputs.pt'],
        'processes':3,'prefix_observations':1500})
repo=Path(firstenv['repo'])
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==PIN
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for path,h in firstenv['source_sha256'].items(): assert sha(repo/path)==h
assert sha(ROOT/'runner/measure_cp4_event.py')==firstenv['runner_sha256']
baseline={'schema_version':1,'baseline_id':ID,'status':'FROZEN','created_utc':datetime.now(timezone.utc).isoformat(),
    'units':'microseconds','source_commit':PIN,'runner_sha256':firstenv['runner_sha256'],
    'protocol_sha256':sha(ROOT/'PROTOCOL_CN.md'),'source_sha256':firstenv['source_sha256'],
    'scope':{'B':1,'Hq':16,'Hk':16,'Hv':16,'D':128,'dtype':'FP16','state_dtype':'FP32',
        'gate_beta_dtype':'FP32','layout':'TND','algorithm_chunk':64,'P':8,
        'initial_state':True,'final_state':True,'use_cp':True,'kernel_qk_l2norm':False,
        'scale':1/(128**.5),'cu_seqlens_dtype':'int64','rotating_qkv_buffers':4},
    'method':{'name':'sparse_cumulative_event_difference','samples_per_probe_per_process':100,
        'probes':['P0','P1','P2','P3','P4'],'calls_per_graph':400,'probe_stride_calls':4,
        'warmup_calls_before_each_probe':20,'independent_processes':3,
        'stage_formula':'mean(Pi)-mean(Pi-1); median of three process estimates',
        'native_formula':'median of 5 graph batch per-call means per process; median across processes',
        'timing_events_external':True,'gpu_streams':1,'host_dispatch_timed':False,
        'workspace_allocation_timed':False,'jit_timed':False,'ncu_used_for_latency':False,
        'fixed_clock':False,'cache_protocol':'natural K1→K2→K3→K4; rotating QKV; no prime/flush',
        'is_direct_single_kernel_event_interval':False},
    'environment':{k:firstenv[k] for k in ['torch','cuda','nvidia-cutlass-dsl','cuda-python','gpu_name','sm_count']},
    'shapes':shapes,'ablation_policy':{
        'absolute_baseline':'Use these frozen stage_us/native_chain_us, do not substitute a same-run B0.',
        'drift_gate_pct':5,'same_run_B0_role':'Environment guard only; >5% needs review/retest.',
        'required_context':'Same full CP chain, inputs SHA, buffer rotation, scope, C, probe placement/statistics.',
        'isolated_K3_old_runs':'Historical diagnostics only; not directly subtractable from this baseline.',
        'immutability':'Never overwrite; new measurements require an explicit new ID.',
        'semantic_breaking_variants':'Report separately; no output-correctness claim.'},
    'limitations':['Stage values are differences of cumulative Event means, not simultaneous direct per-kernel timestamps.',
        'Remaining marker-position/cache/clock bias is not eliminated merely because sums close.',
        'No HBM/L2 traffic measured here; natural cache history is not a guarantee of HBM residency.',
        'Unsegmented graph E2E is GPU-only batch per-call mean, not host public API wall time.',
        'No time→cycle conversion without time-window-matched clock evidence.']}
with (ROOT/'frozen_baseline.json').open('x') as f:json.dump(baseline,f,ensure_ascii=False,indent=2);f.write('\n')
with (ROOT/'validated_process_data.json').open('x') as f:json.dump({'baseline':baseline,'processes':allruns,'checks':check},f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'baseline_id':ID,'status':'FROZEN','shapes':shapes,'sha256':sha(ROOT/'frozen_baseline.json')},ensure_ascii=False),flush=True)
