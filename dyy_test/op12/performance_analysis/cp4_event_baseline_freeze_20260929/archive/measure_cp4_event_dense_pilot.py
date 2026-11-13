#!/usr/bin/env python3
"""Unmodified public CP API, contiguous four-stage graph Event measurement."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback

PIN = 'e77a4a0d276367895c3b50a642fd8f326c03fb72'
BASELINE_ID = 'OP12_CP4_EVENT_FP16_B1H16_P8_20260929_V1'
STAGES = [
    ('K1', 'cp_delta_rule_t_precompute_dsl_sm100'),
    ('K2', 'cp_delta_rule_mn_precompute_dsl_sm100'),
    ('K3', 'cp_delta_rule_fixup_dsl_sm100'),
    ('K4', 'cp_delta_rule_prefill_dsl_sm100'),
]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def command(args):
    return subprocess.check_output(args, text=True).strip()


def telemetry_sample():
    query = 'timestamp,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used'
    return command(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'])


def percentile(xs, p):
    ys = sorted(xs)
    t = (len(ys)-1)*p
    lo, hi = math.floor(t), math.ceil(t)
    return ys[lo] + (ys[hi]-ys[lo])*(t-lo)


def stats(xs):
    mean = statistics.mean(xs)
    return {'n': len(xs), 'median_us': statistics.median(xs),
            'mean_us': mean, 'p5_us': percentile(xs, .05),
            'p95_us': percentile(xs, .95),
            'cv_pct': statistics.pstdev(xs)/mean*100,
            'min_us': min(xs), 'max_us': max(xs)}


def inspect_dot(path, repeats, staged):
    s = Path(path).read_text()
    nodes=list(re.finditer(r'"(graph_\d+_node_\d+)"\[(.*?)\];',s,re.S))
    kernels=[]
    event_count=0
    for m in nodes:
        body=m.group(2)
        if 'EVENT_RECORD' in body: event_count+=1
        if 'label="{KERNEL' not in body: continue
        line=body.splitlines()[1]
        fields=line.split(' | ',2)
        launch=fields[-1].rsplit('\\<\\<\\<',1)
        config=launch[1].split('\\>')[0].split(',')
        kernels.append({'name':launch[0].strip(), 'grid':int(config[0]),
                        'block':int(config[1]),'dynamic_smem_bytes':int(config[2]),'node':m.group(1)})
    expected = 4*repeats
    if len(kernels) != expected:
        raise RuntimeError(f'DOT kernel count {len(kernels)} != {expected}: {path}')
    first = kernels[:4]
    def signature(k): return {a:b for a,b in k.items() if a!='node'}
    for i, kernel in enumerate(kernels):
        if signature(kernel) != signature(first[i % 4]):
            raise RuntimeError(f'DOT kernel signature changed at launch {i}')
    expected_events=5*repeats+2 if staged else 2
    assert event_count==expected_events,(event_count,expected_events)
    edges=re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"',s)
    expected_edges=list(zip([x.group(1) for x in nodes][:-1],[x.group(1) for x in nodes][1:]))
    assert set(edges)==set(expected_edges),'Graph is not one strictly serialized node chain'
    assert [k['block'] for k in first]==[128,384,256,384]
    assert [k['grid'] for k in first[1:]]==[128,32,128]
    return {'kernel_nodes': len(kernels), 'four_kernel_sequence_repetitions': repeats,
            'event_nodes':event_count,'all_nodes_single_serial_chain':True,
            'stage_kernel_labels': dict(zip([x[0] for x in STAGES], first)),
            'dot_sha256': sha_file(path), 'staged': staged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--s', type=int, choices=[8192,16384,32768], required=True)
    ap.add_argument('--trial', type=int, required=True)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--native-repeats', type=int, default=5)
    ap.add_argument('--pilot', action='store_true')
    args = ap.parse_args()
    out = Path(args.output_root)/('pilot' if args.pilot else 'formal')/f'S{args.s}_trial{args.trial}'
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUTE_DSL_DUMP_DIR', str(out/'compiler_dump'))
    os.environ.setdefault('CUTE_DSL_KEEP_PTX', '1')
    os.environ.setdefault('CUTE_DSL_KEEP_CUBIN', '1')
    repo = Path(args.repo).resolve()
    assert command(['git','-C',str(repo),'rev-parse','HEAD']) == PIN
    assert not command(['git','-C',str(repo),'status','--porcelain'])
    sys.path.insert(0, str(repo))
    import torch
    import torch.nn.functional as F
    import flashinfer
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host
    assert Path(flashinfer.__file__).resolve().is_relative_to(repo)
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    assert props.multi_processor_count == 148 and props.name == 'NVIDIA B200'
    c = choose_cp_chunk_len_host(args.s,16,148,device_capability=(10,0),
                                total_seqlen=args.s,num_seqs=1,device_name=props.name)
    assert c == {8192:1024,16384:2048,32768:4096}[args.s]
    seed = 20260929 + args.s
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    telemetry = [{'phase':'start','wall_ns':time.time_ns(),'nvidia_smi':telemetry_sample()}]
    env = {'baseline_id':BASELINE_ID,'pilot':args.pilot,'S':args.s,'C':c,'P':8,'B':1,'H':16,'D':128,
           'trial':args.trial,'pid':os.getpid(),'seed':seed,'rotating_buffers':4,
           'warmup':args.warmup,'samples':args.samples,'pin':PIN,'repo':str(repo),
           'flashinfer_import':flashinfer.__file__,'torch':torch.__version__,'cuda':torch.version.cuda,
           'gpu_name':props.name,'sm_count':props.multi_processor_count,
           'compute_capability':list(torch.cuda.get_device_capability()),
           'memory_bytes':props.total_memory,'telemetry':telemetry,
           'runner_sha256':sha_file(__file__),'cu_seqlens_dtype':'torch.int64',
           'initial_final_state_dtype':'torch.float32','qkvo_dtype':'torch.float16',
           'kernel_qk_l2norm':False,'use_cp':True,'scale':1/math.sqrt(128)}
    for dist in ['nvidia-cutlass-dsl','cuda-python','flashinfer-python']:
        try: env[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: env[dist] = None
    source_paths = ['flashinfer/gdn_prefill.py',
        'flashinfer/gdn_kernels/blackwell/gdn_cp_prefill.py',
        'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp.py',
        'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp_prefill.py',
        'flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py',
        'flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py']
    env['source_sha256'] = {p:sha_file(repo/p) for p in source_paths}
    write_json(out/'environment.json', env)

    # Generate once on CPU then retain exact inputs for future matching ablations.
    cpu_inputs = []
    for i in range(4):
        q = torch.randn(args.s,16,128,dtype=torch.float16)
        k = F.normalize(torch.randn(args.s,16,128,dtype=torch.float32),p=2,dim=-1).to(torch.float16)
        v = torch.randn(args.s,16,128,dtype=torch.float16)
        cpu_inputs.append({'q':q,'k':k,'v':v})
    shared = {'gate':torch.rand(args.s,16,dtype=torch.float32),
              'beta':torch.rand(args.s,16,dtype=torch.float32).sigmoid(),
              'initial':torch.randn(1,16,128,128,dtype=torch.float32)*.01,
              'cu_seqlens':torch.tensor([0,args.s],dtype=torch.int64)}
    input_manifest = {'seed':seed,'buffers':[], 'shared':{}}
    for b in cpu_inputs:
        input_manifest['buffers'].append({k:{'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest(),
            'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for k,v in b.items()})
    for k,v in shared.items():
        input_manifest['shared'][k]={'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest(),
            'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()}
    input_manifest['content_signature_sha256'] = hashlib.sha256(
        json.dumps(input_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(out/'inputs_manifest.json', input_manifest)
    if not args.pilot and args.trial == 1:
        inp_dir = Path(args.output_root)/'frozen_inputs'/f'S{args.s}'
        inp_dir.mkdir(parents=True, exist_ok=False)
        torch.save({'buffers':cpu_inputs,'shared':shared}, inp_dir/'inputs.pt')
        write_json(inp_dir/'inputs_manifest.json', input_manifest)
        write_json(inp_dir/'storage_hash.json', {'inputs.pt':sha_file(inp_dir/'inputs.pt')})
    buffers = [{**{k:v.to(device) for k,v in b.items()},
                'output':torch.empty(args.s,16,128,dtype=torch.float16,device=device),
                'output_state':torch.empty(1,16,128,128,dtype=torch.float32,device=device)} for b in cpu_inputs]
    gpu_shared = {k:v.to(device) for k,v in shared.items()}
    del cpu_inputs
    initial_copy = gpu_shared['initial'].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke(i, use_cp=True, output=None, state=None):
        b = buffers[i]
        return chunk_gated_delta_rule(b['q'],b['k'],b['v'],g=gpu_shared['gate'],beta=gpu_shared['beta'],
            scale=1/math.sqrt(128),initial_state=gpu_shared['initial'],output_final_state=True,
            cu_seqlens=gpu_shared['cu_seqlens'],use_qk_l2norm_in_kernel=False,
            output=b['output'] if output is None else output,
            output_state=b['output_state'] if state is None else state,use_cp=use_cp)

    print(json.dumps({'phase':'JIT_and_correctness','S':args.s,'trial':args.trial}), flush=True)
    with torch.cuda.stream(stream):
        for i in range(4): invoke(i)
    stream.synchronize()
    reference = [(b['output'].clone(),b['output_state'].clone()) for b in buffers]
    torch.cuda.synchronize()
    validation = {'cp_graph_exact': [], 'non_cp_comparison': {}, 'initial_unchanged':None}
    no = torch.empty_like(buffers[0]['output'])
    ns = torch.empty_like(buffers[0]['output_state'])
    with torch.cuda.stream(stream): invoke(0,use_cp=False,output=no,state=ns)
    stream.synchronize()
    for name,a,b in [('O',reference[0][0],no),('final_state',reference[0][1],ns)]:
        af,bf=a.float(),b.float()
        validation['non_cp_comparison'][name]={'atol':.04,'rtol':.04,
            'allclose':torch.allclose(a,b,atol=.04,rtol=.04),
            'relative_l2':((af-bf).norm()/bf.norm().clamp_min(1e-12)).item(),
            'max_abs':(af-bf).abs().max().item()}
    if not all(x['allclose'] for x in validation['non_cp_comparison'].values()):
        raise RuntimeError(f'CP/nonCP validation failed: {validation}')

    # Preallocate and initialize every CUDA Event before stream capture.
    stage_events = [[torch.cuda.Event(enable_timing=True,external=True) for _ in range(5)]
                    for _ in range(args.samples)]
    native_events = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
    batch_events = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
    with torch.cuda.stream(stream):
        for evs in stage_events:
            for ev in evs: ev.record(stream)
        for ev in native_events + batch_events: ev.record(stream)
    stream.synchronize()
    stage_trace=[]
    workspace_snap={}
    current_sample = [-1]
    original = {name:getattr(cp,name) for _,name in STAGES}

    @contextlib.contextmanager
    def boundaries():
        def make(stage_idx,name):
            def wrapped(*a,**kw):
                if stage_idx == 0:
                    current_sample[0] += 1
                    stage_events[current_sample[0]][0].record(stream)
                ret=original[name](*a,**kw)
                stage_events[current_sample[0]][stage_idx+1].record(stream)
                stage_trace.append({'sample':current_sample[0],'stage':stage_idx+1})
                if current_sample[0] == 0:
                    tensors = list(ret) if isinstance(ret,tuple) else [ret]
                    workspace_snap[STAGES[stage_idx][0]] = [
                        {'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size(),
                         'address':t.data_ptr()} for t in tensors if isinstance(t,torch.Tensor)]
                return ret
            return wrapped
        try:
            for idx,(_,name) in enumerate(STAGES): setattr(cp,name,make(idx,name))
            yield
        finally:
            for name,fn in original.items(): setattr(cp,name,fn)

    staged_graph=torch.cuda.CUDAGraph(keep_graph=True)
    staged_graph.enable_debug_mode()
    with boundaries(), torch.cuda.graph(staged_graph,stream=stream):
        batch_events[0].record(stream)
        for j in range(args.samples): invoke(j%4)
        batch_events[1].record(stream)
    staged_graph.debug_dump(str(out/'staged_graph.dot'))
    staged_graph.instantiate()
    native_graph=torch.cuda.CUDAGraph(keep_graph=True)
    native_graph.enable_debug_mode()
    with torch.cuda.graph(native_graph,stream=stream):
        native_events[0].record(stream)
        for j in range(args.samples): invoke(j%4)
        native_events[1].record(stream)
    native_graph.debug_dump(str(out/'native_graph.dot'))
    native_graph.instantiate()
    assert stage_trace == [{'sample':i,'stage':k} for i in range(args.samples) for k in range(1,5)]
    graph_validation={'staged':inspect_dot(out/'staged_graph.dot',args.samples,True),
                      'native':inspect_dot(out/'native_graph.dot',args.samples,False),
                      'capture_stage_sequence':stage_trace,'workspace_metadata':workspace_snap}
    write_json(out/'graph_validation.json',graph_validation)

    stop=threading.Event()
    def monitor():
        while not stop.wait(.2):
            try: telemetry.append({'phase':'measurement','wall_ns':time.time_ns(),'nvidia_smi':telemetry_sample()})
            except Exception as e: telemetry.append({'monitor_error':str(e)})
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    print(json.dumps({'phase':'measurement','S':args.s,'trial':args.trial,
                      'kernel_nodes':4*args.samples}),flush=True)
    with torch.cuda.stream(stream):
        staged_graph.replay()  # Graph warmup, excluded.
        for j in range(args.warmup): invoke(j%4)
    stream.synchronize()
    telemetry.append({'phase':'before_staged','wall_ns':time.time_ns(),'nvidia_smi':telemetry_sample()})
    with torch.cuda.stream(stream): staged_graph.replay()
    stream.synchronize()
    staged_batch_per_call = batch_events[0].elapsed_time(batch_events[1])*1000/args.samples
    rows=[]
    for i,evs in enumerate(stage_events):
        vals=[evs[j].elapsed_time(evs[j+1])*1000 for j in range(4)]
        total=evs[0].elapsed_time(evs[4])*1000
        assert all(x>0 for x in vals) and abs(sum(vals)-total) < .01
        rows.append({'S':args.s,'C':c,'trial':args.trial,'sample':i,'buffer':i%4,
                     **dict(zip(['K1_us','K2_us','K3_us','K4_us'],vals)),'chain_us':total})
    with open(out/'stage_samples.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    native=[]
    with torch.cuda.stream(stream): native_graph.replay()
    stream.synchronize()
    for r in range(args.native_repeats):
        with torch.cuda.stream(stream): native_graph.replay()
        stream.synchronize()
        native.append({'repeat':r,'batch_us':native_events[0].elapsed_time(native_events[1])*1000,
                       'per_call_mean_us':native_events[0].elapsed_time(native_events[1])*1000/args.samples})
    write_json(out/'native_batch_samples.json',native)
    stop.set();thread.join(timeout=5)
    telemetry.append({'phase':'end','wall_ns':time.time_ns(),'nvidia_smi':telemetry_sample()})

    for i,b in enumerate(buffers):
        o,s=b['output'],b['output_state']
        ck={'buffer':i,'O_exact':torch.equal(o,reference[i][0]),
            'state_exact':torch.equal(s,reference[i][1]),
            'O_finite':bool(torch.isfinite(o).all()),'state_finite':bool(torch.isfinite(s).all())}
        validation['cp_graph_exact'].append(ck)
        assert all(ck[k] for k in ['O_exact','state_exact','O_finite','state_finite']),ck
    validation['initial_unchanged']=torch.equal(initial_copy,gpu_shared['initial'])
    assert validation['initial_unchanged']
    write_json(out/'validation.json',validation)
    env['telemetry']=telemetry
    write_json(out/'environment.json',env)
    summary={key:stats([r[key] for r in rows]) for key in ['K1_us','K2_us','K3_us','K4_us','chain_us']}
    native_mean=statistics.median(x['per_call_mean_us'] for x in native)
    summary['native_batch_per_call'] = stats([x['per_call_mean_us'] for x in native])
    summary['staged_batch_per_call_mean_us']=staged_batch_per_call
    summary['marker_check_pct']=(staged_batch_per_call/native_mean-1)*100
    summary['baseline_id']=BASELINE_ID
    summary['input_signature']=input_manifest['content_signature_sha256']
    summary['status']='PASS'
    write_json(out/'summary.json',summary)
    assert not command(['git','-C',str(repo),'status','--porcelain'])
    print(json.dumps({'phase':'complete','S':args.s,'trial':args.trial,'summary':summary}),flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
