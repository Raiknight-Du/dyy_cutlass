#!/usr/bin/env python3
"""Read-only ablation comparison. Frozen point stays fixed; B0 is only a guard."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--baseline',type=Path,required=True)
ap.add_argument('--variant-dir',type=Path,required=True)
ap.add_argument('--same-round-b0-dir',type=Path,required=True)
args=ap.parse_args()
load=lambda p:json.loads(p.read_text())
base=load(args.baseline)
assert base['status']=='FROZEN'

def context(folder,control):
    env,su,inp,gv=[load(folder/n) for n in ['environment.json','summary.json','inputs_manifest.json','graph_validation.json']]
    shape=next(x for x in base['shapes'] if x['S']==env['S'])
    expected={'pin':base['source_commit'],'B':1,'H':16,'D':128,'C':shape['C'],'P':8,
        'rotating_buffers':4,'warmup':20,'samples':100,'calls_per_graph':400,'probe_stride_calls':4,
        'qkvo_dtype':'torch.float16','initial_final_state_dtype':'torch.float32',
        'cu_seqlens_dtype':'torch.int64','kernel_qk_l2norm':False,'use_cp':True,'scale':base['scope']['scale']}
    for key,value in expected.items():
        if env.get(key)!=value:raise ValueError(f'{folder}: incompatible context {key}')
    for key in ['torch','cuda','nvidia-cutlass-dsl','cuda-python','gpu_name','sm_count']:
        if env.get(key)!=base['environment'][key]:raise ValueError(f'{folder}: environment mismatch {key}')
    if inp['content_signature_sha256']!=shape['input_signature_sha256'] or su['input_signature']!=shape['input_signature_sha256']:
        raise ValueError(f'{folder}: input SHA mismatch')
    if set(gv['probes'])!={'P0','P1','P2','P3','P4'}:raise ValueError('Not the frozen five-probe method')
    for graph in list(gv['probes'].values())+[gv['native']]:
        if graph['kernel_nodes']!=1600 or not graph['all_nodes_single_serial_chain']:
            raise ValueError('Not 400 real four-kernel calls in a serial chain')
        if {k:v['grid'] for k,v in graph['stage_kernel_labels'].items()}!=shape['grid']:
            raise ValueError('Grid mismatch')
    if any(p['event_nodes']!=202 for p in gv['probes'].values()) or gv['native']['event_nodes']!=2:
        raise ValueError('Event placement/count mismatch')
    if control and env['source_sha256']!=base['source_sha256']:
        raise ValueError('B0 is not the pinned unmodified upstream source')
    for i in range(1,5):
        if su['stage_estimates'][f'K{i}']['method']!=f'mean(P{i})-mean(P{i-1})':
            raise ValueError('Statistics convention mismatch')
    return shape,su

try:
    shape,v=context(args.variant_dir,False)
    shape0,b0=context(args.same_round_b0_dir,True)
    if shape0['S']!=shape['S']:raise ValueError('B0/variant shape mismatch')
    result={'baseline_id':base['baseline_id'],'baseline_file_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'S':shape['S'],'comparison':{},'drift':{},'causal_admission':'PASS'}
    for k,frozen in shape['stage_us'].items():
        obs=v['stage_estimates'][k]['estimate_us']
        drift=(b0['stage_estimates'][k]['estimate_us']/frozen-1)*100
        result['comparison'][k]={'frozen_us':frozen,'variant_us':obs,
            'saved_us':frozen-obs,'speedup':frozen/obs,
            'baseline_process_range_us':shape['stage_process_minmax_us'][k]}
        result['drift'][k]=drift
    e2e=shape['native_chain_us']
    result['comparison']['native_chain']={'frozen_us':e2e,'variant_us':v['native_batch_per_call']['median_us'],
        'saved_us':e2e-v['native_batch_per_call']['median_us'],'speedup':e2e/v['native_batch_per_call']['median_us']}
    result['drift']['native_chain']=(b0['native_batch_per_call']['median_us']/e2e-1)*100
    if any(abs(x)>base['ablation_policy']['drift_gate_pct'] for x in result['drift'].values()):
        result['causal_admission']='REVIEW_REQUIRED_DRIFT'
    result['warning']='Do not replace frozen baseline with same-round B0. Small changes need uncertainty/replication checks; stage estimates have residual bias.'
    print(json.dumps(result,ensure_ascii=False,indent=2))
except (ValueError,KeyError,StopIteration,AssertionError) as exc:
    print(json.dumps({'status':'INCOMPATIBLE','reason':str(exc)},ensure_ascii=False))
    sys.exit(2)
