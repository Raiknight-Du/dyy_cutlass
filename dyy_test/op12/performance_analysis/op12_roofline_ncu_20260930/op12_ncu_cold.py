#!/usr/bin/env python3
"""One complete FlashInfer CP graph execution after a read-only L2 eviction."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


OP12 = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12')
REPO = OP12 / 'upstreams/flashinfer'
FROZEN = OP12 / 'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs'
PIN = 'e77a4a0d276367895c3b50a642fd8f326c03fb72'
STAGES = ('CPDeltaRuleTPrecomputeSm100', 'CPDeltaRuleMNPrecomputeUtcmma1Sm100',
          'CPDeltaRuleFixupUtcmmaSm100', 'CPDeltaRulePrefillTcgen05Sm100')
CHUNKS = {8192: 1024, 16384: 2048, 32768: 4096}


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')


def check_graph(dot_path):
    text = dot_path.read_text()
    kernels = re.findall(r'"(graph_\d+_node_\d+)"\[.*?label="\{KERNEL(.*?)\}\"];', text, re.S)
    if len(kernels) != 4:
        raise AssertionError(f'expected exactly four graph kernel nodes, found {len(kernels)}')
    for index, ((_, label), stage) in enumerate(zip(kernels, STAGES)):
        if stage not in label:
            raise AssertionError(f'graph stage {index+1}: expected {stage}, found {label[:180]}')
    nodes = [node for node, _ in kernels]
    edges = set(re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', text))
    if len(nodes) != 4 or any((a, b) not in edges for a, b in zip(nodes, nodes[1:])):
        raise AssertionError('graph kernels are not one ordered K1->K4 chain')
    return {'kernel_count': 4, 'stages': list(STAGES), 'nodes': nodes,
            'edges': sorted([list(edge) for edge in edges]), 'sha256': digest(dot_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=int, choices=CHUNKS, required=True)
    parser.add_argument('--sample', type=int, required=True)
    parser.add_argument('--evict-factor', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--post-evict-idle-ms', type=int, default=0)
    parser.add_argument('--tail-mode', choices=('none', 'after_cp', 'control'), default='none')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out / f'app_pass_{os.getpid()}_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)

    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == PIN, commit
    assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
    source_paths = ('flashinfer/gdn_prefill.py',
                    'flashinfer/gdn_kernels/blackwell/gdn_cp_prefill.py',
                    'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp.py',
                    'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp_prefill.py')
    source_hashes = {name: digest(REPO / name) for name in source_paths}
    sys.path.insert(0, str(REPO))
    import torch
    import flashinfer
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host

    assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
    torch.set_num_threads(1)
    torch.manual_seed(20260929 + args.shape)
    torch.cuda.manual_seed_all(20260929 + args.shape)
    props = torch.cuda.get_device_properties(0)
    assert props.name == 'NVIDIA B200' and props.multi_processor_count == 148
    assert choose_cp_chunk_len_host(args.shape, 16, 148,
        device_capability=(10, 0), total_seqlen=args.shape, num_seqs=1,
        device_name=props.name) == CHUNKS[args.shape]
    assert props.L2_cache_size > 0

    frozen = FROZEN / f'S{args.shape}' / 'inputs.pt'
    cpu = torch.load(frozen, map_location='cpu', weights_only=True)
    assert len(cpu['buffers']) == 4
    device = torch.device('cuda:0')
    qkv = {name: cpu['buffers'][0][name].to(device) for name in ('q', 'k', 'v')}
    shared = {name: tensor.to(device) for name, tensor in cpu['shared'].items()}
    output = torch.empty((args.shape, 16, 128), dtype=torch.float16, device=device)
    final_state = torch.empty((1, 16, 128, 128), dtype=torch.float32, device=device)
    initial_copy = shared['initial'].clone()
    del cpu
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke():
        return chunk_gated_delta_rule(qkv['q'], qkv['k'], qkv['v'],
            g=shared['gate'], beta=shared['beta'], scale=1/math.sqrt(128),
            initial_state=shared['initial'], output_final_state=True,
            cu_seqlens=shared['cu_seqlens'], use_qk_l2norm_in_kernel=False,
            output=output, output_state=final_state, use_cp=True)

    # JIT, allocation, full-CP warmup, and reference all precede eviction.
    with torch.cuda.stream(stream):
        invoke()
    stream.synchronize()
    reference_o, reference_state = output.clone(), final_state.clone()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    with torch.cuda.graph(graph, stream=stream):
        invoke()
    dot_path = out / 'cp4_graph.dot'
    graph.debug_dump(str(dot_path))
    graph.instantiate()
    graph_info = check_graph(dot_path)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.equal(output, reference_o) and torch.equal(final_state, reference_state)

    # Materialize non-compressible, independent read-only eviction data now.
    evict_bytes = props.L2_cache_size * args.evict_factor
    evict = torch.rand((evict_bytes // 4,), dtype=torch.float32, device=device)
    assert evict.numel() * evict.element_size() >= evict_bytes
    smi_query = ['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits']
    telemetry_before = subprocess.check_output(smi_query, text=True).strip()
    with torch.cuda.stream(stream):
        output.zero_()
        final_state.zero_()
    stream.synchronize()
    # Last GPU activity before K1. The scalar is consumed only after profiling.
    with torch.cuda.stream(stream):
        evict_checksum = evict.sum()
    stream.synchronize()
    if args.post_evict_idle_ms:
        time.sleep(args.post_evict_idle_ms / 1000)
    print(json.dumps({'event': 'measured_graph_start', 'shape': args.shape,
                      'sample': args.sample, 'evict_factor': args.evict_factor,
                      'l2_bytes': props.L2_cache_size,
                      'evict_bytes': evict.numel() * evict.element_size()}), flush=True)
    range_name = {'none': 'OP12_CP4_COLD', 'after_cp': 'OP12_CP4_WITH_TAIL',
                  'control': 'OP12_CP4_TAIL_CONTROL'}[args.tail_mode]
    torch.cuda.nvtx.range_push(range_name)
    with torch.cuda.stream(stream):
        if args.tail_mode != 'control':
            graph.replay()
        tail_checksum = evict.sum() if args.tail_mode != 'none' else None
    torch.cuda.nvtx.range_pop()
    stream.synchronize()
    telemetry_after = subprocess.check_output(smi_query, text=True).strip()

    checks = {'output_exact': args.tail_mode == 'control' or torch.equal(output, reference_o),
              'state_exact': args.tail_mode == 'control' or torch.equal(final_state, reference_state),
              'output_finite': torch.isfinite(output).all().item(),
              'state_finite': torch.isfinite(final_state).all().item(),
              'initial_unchanged': torch.equal(shared['initial'], initial_copy),
              'eviction_checksum_finite': math.isfinite(evict_checksum.item()),
              'tail_checksum_finite': tail_checksum is None or math.isfinite(tail_checksum.item())}
    assert all(checks.values()), checks
    output_hashes = {
        'output_sha256': hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest(),
        'final_state_sha256': hashlib.sha256(final_state.detach().cpu().numpy().tobytes()).hexdigest(),
    }
    manifest = {'shape': args.shape, 'chunk': CHUNKS[args.shape], 'sample': args.sample,
        'evict_factor': args.evict_factor, 'l2_bytes': props.L2_cache_size,
        'evict_bytes': evict.numel() * evict.element_size(), 'graph': graph_info,
        'post_evict_idle_ms': args.post_evict_idle_ms,
        'tail_mode': args.tail_mode,
        'source_commit': commit, 'flashinfer_import': flashinfer.__file__,
        'source_sha256': source_hashes,
        'frozen_input_file': str(frozen), 'frozen_input_sha256': digest(frozen),
        'harness_sha256': digest(Path(__file__)), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'gpu_name': props.name,
        'gpu_sm_count': props.multi_processor_count,
        'telemetry_before': telemetry_before, 'telemetry_after': telemetry_after,
        'input_dtypes': {'qkv_output': 'float16', 'gate_beta_state': 'float32',
                         'cu_seqlens': 'int64'},
        'public_entry': 'flashinfer.gdn_prefill.chunk_gated_delta_rule',
        'use_cp': True, 'output_final_state': True,
        'use_qk_l2norm_in_kernel': False, 'scale': 1/math.sqrt(128),
        'checks': checks, 'output_hashes': output_hashes}
    save(out / 'pass_manifest.json', manifest)
    print(json.dumps({'event': 'app_pass_complete', 'manifest': str(out / 'pass_manifest.json'),
                      'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
