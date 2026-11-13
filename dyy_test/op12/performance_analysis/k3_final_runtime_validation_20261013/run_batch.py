"""Runtime-mode same-object tests; no IKET and no public API modifications."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
ROOT = Path(__file__).resolve().parent
OP12 = ROOT.parents[1]
AUDIT = ROOT.parent

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m

def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)

def spec(v):
    return {'variant': v, 'id': v + '_M2N1', 'm_stage': 2, 'n_stage': 1}

def graph_for(torch, fn, runtime, kwargs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt = list(runtime)
        rt[-1] = type(rt[-1])(torch.cuda.current_stream().cuda_stream)
        for _repeat in range(64):
            fn(*rt, **kwargs)
    torch.cuda.synchronize()
    return graph

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'event', 'profile', 'slope', 'micro_profile'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--style', choices=['RUNTIME', 'COALESCED'], default='RUNTIME')
    p.add_argument('--mode', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--repeats', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument('--samples', type=int, default=100)
    args = p.parse_args()
    if args.phase in ('profile', 'micro_profile'):
        args.out = args.out / ('pid_' + str(os.getpid()))
    args.case = (32768, 4096, 16)
    args.cache = 'HOT'
    args.fixture = 'REAL_PREFIX'
    args.repo = OP12 / 'upstreams/flashinfer'
    args.helper = AUDIT / 'k3_iket_stage_analysis_20260928/implementation/run_k3_iket.py'
    args.implementation = ROOT / 'runtime_k3.py'
    base = load(AUDIT / 'k3_causal_ablation_20260928/runner/run_causal_k3.py', 'runtime_frozen_host')
    base.check_exclusive_gpu()
    w = base.Workload(args)
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as driver

    def compile_private(style):
        cls = w.generated.get_kernel_class(style)
        w.prefill._get_fixup_kernel.cache_clear()
        w.prefill.CPDeltaRuleFixupUtcmmaSm100 = cls
        w.active_binary[0] = w.out / 'binary' / style
        old_get = w.prefill.get_cached_compile
        old_compile = w.prefill.cached_compile
        captured = {}

        def compile_and_defer(kernel, *rt, **kw):
            rt = rt[:-1] + (cutlass.Int32(3), rt[-1])
            compiled = old_compile(kernel, *rt, **kw)
            captured['compiled'] = compiled

            def hold(*call, **kwargs):
                captured['runtime_args'] = call[:-1] + (cutlass.Int32(3), call[-1])
                captured['runtime_kwargs'] = kwargs
            return hold
        w.prefill.get_cached_compile = lambda *a: None
        w.prefill.cached_compile = compile_and_defer
        try:
            fixed = w.helper.call_fixup(w.prefill, w.m, w.n, w.buffers, w.S, w.C, w.torch, 'PRIVATE_COMPILE')
        finally:
            w.prefill.get_cached_compile = old_get
            w.prefill.cached_compile = old_compile
            w.prefill.CPDeltaRuleFixupUtcmmaSm100 = w.original_class
            w.prefill._get_fixup_kernel.cache_clear()
        record = {'key': style, 'spec': spec(style), 'captured': captured, 'fixed': fixed, 'binary_dir': w.active_binary[0], 'level': 'event', 'class_name': cls.__name__}
        assert len(captured['runtime_args']) == 14
        w.compiled[style] = record
        return record

    def binding(record, mode):
        (fn, rt, kw) = w.bind(record)
        rt = list(rt)
        rt[-2] = cutlass.Int32(mode)
        return (fn, tuple(rt), kw)
    torch = w.torch
    try:
        upstream = w.compile_spec(spec('B0'), original=True)
        w.launch(upstream)
        torch.cuda.synchronize()
        reference = (upstream['fixed'].cpu(), w.buffers['final'].cpu())
        records = {s: compile_private(s) for s in ['RUNTIME', 'COALESCED']}
        calls = {}
        checks = []
        for (style, r) in records.items():
            for mode in (1, 2, 3):
                (fn, rt, kw) = binding(r, mode)
                calls[style, mode] = (fn, rt, kw)
                r['fixed'].fill_(-12345)
                w.buffers['final'].fill_(-12345)
                fn(*rt, **kw)
                torch.cuda.synchronize()
                err_fixed = float((r['fixed'].cpu() - reference[0]).abs().max())
                err_final = None if mode == 1 else float((w.buffers['final'].cpu() - reference[1]).abs().max())
                sentinel = bool((w.buffers['final'] == -12345).all()) if mode == 1 else True
                check = {'style': style, 'mode': mode, 'fixed_max_abs': err_fixed, 'final_max_abs': err_final, 'sentinel': sentinel, 'passed': err_fixed <= 1e-06 and (err_final is None or err_final <= 1e-06) and sentinel}
                checks.append(check)
                assert check['passed'], check
        write(w.out / 'correctness.json', checks)
        (fn0, rt0, kw0) = w.bind(upstream)
        calls['UPSTREAM', 3] = (fn0, rt0, kw0)
        if args.phase == 'profile':
            w.prepare(upstream)
            (fn, rt, kw) = calls[args.style, args.mode]
            torch.cuda.nvtx.range_push('K3_RUNTIME_TARGET')
            fn(*rt, **kw)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        elif args.phase in ('smoke', 'event'):
            graphs = {k: graph_for(torch, *call) for (k, call) in calls.items()}
            for (k, g) in graphs.items():
                if k[0] == 'UPSTREAM':
                    continue
                records[k[0]]['fixed'].fill_(-12345)
                g.replay()
                torch.cuda.synchronize()
                assert float((records[k[0]]['fixed'].cpu() - reference[0]).abs().max()) <= 1e-06
            if args.phase == 'event':
                for (k, call) in calls.items():
                    for _ in range(20):
                        w.prepare(upstream)
                        call[0](*call[1], **call[2])
                torch.cuda.synchronize()
                rows = []
                rng = random.Random(1310 + args.trial)
                targets = [k for k in calls if k != ('UPSTREAM', 3)]
                for method in ('direct', 'graph_batch64'):
                    for pair in range(args.samples):
                        if pair % 10 == 0:
                            base.check_exclusive_gpu()
                        rng.shuffle(targets)
                        for target in targets:
                            for (label, baseline) in [('MODE3', (target[0], 3)), ('UPSTREAM', ('UPSTREAM', 3))]:
                                arms = [('BASE', baseline), ('VAR', target)]
                                if (pair + args.trial) % 2:
                                    arms.reverse()
                                for (arm, key) in arms:
                                    w.prepare(upstream)
                                    (begin, end) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                    begin.record()
                                    if method == 'direct':
                                        (fn, rt, kw) = calls[key]
                                        fn(*rt, **kw)
                                    else:
                                        graphs[key].replay()
                                    end.record()
                                    end.synchronize()
                                    rows.append({'trial': args.trial, 'method': method, 'comparison': target[0] + '_M' + str(target[1]), 'reference': label, 'pair': pair, 'arm': arm, 'us': begin.elapsed_time(end) * 1000 / (64 if method != 'direct' else 1)})
                with (w.out / 'event_samples.csv').open('x', newline='') as f:
                    cw = csv.DictWriter(f, fieldnames=list(rows[0]))
                    cw.writeheader()
                    cw.writerows(rows)
                summary = []
                for method in ('direct', 'graph_batch64'):
                    for target in targets:
                        for label in ['MODE3', 'UPSTREAM']:
                            selected = [x for x in rows if x['method'] == method and x['comparison'] == target[0] + '_M' + str(target[1]) and (x['reference'] == label)]
                            arms = {a: {r['pair']: r['us'] for r in selected if r['arm'] == a} for a in ('BASE', 'VAR')}
                            summary.append({'method': method, 'target': list(target), 'reference': label, 'base': base.distribution(list(arms['BASE'].values())), 'variant': base.distribution(list(arms['VAR'].values())), 'paired_median_gain_us': statistics.median((arms['BASE'][i] - arms['VAR'][i] for i in arms['BASE']))})
                write(w.out / 'event_summary.json', summary)
        else:
            from final_helper import FinalHelper
            initial = w.buffers['final'].detach().clone().reshape(32, 64, 128)
            fixed = torch.empty((8, 32, 64, 128), device='cuda', dtype=torch.float32)
            final = torch.empty_like(fixed)
            views = [from_dlpack(initial.permute(1, 2, 0), assumed_align=16, enable_tvm_ffi=True), from_dlpack(fixed.permute(2, 3, 0, 1), assumed_align=16, enable_tvm_ffi=True), from_dlpack(final.permute(2, 3, 0, 1), assumed_align=16, enable_tvm_ffi=True)]
            stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            micros = {}
            microchecks = []
            for style in ['RUNTIME', 'COALESCED']:
                w.active_binary[0] = w.out / 'binary' / ('MICRO_' + style)
                obj = FinalHelper(coalesced=style == 'COALESCED')
                options = w.prefill._blackwell_compile_options(torch.device('cuda'))
                fn = cute.compile[options](obj, *views, cutlass.Int32(16), cutlass.Int32(3), stream)
                micros[style] = fn
                for mode in (1, 2, 3):
                    fixed.fill_(-12345)
                    final.fill_(-12345)
                    fn(*views, cutlass.Int32(16), cutlass.Int32(mode), stream)
                    torch.cuda.synchronize()
                    fixederr = None if mode == 2 else float((fixed - initial[None]).abs().max())
                    finalerr = None if mode == 1 else float((final - initial[None]).abs().max())
                    omitted = bool((final == -12345).all()) if mode == 1 else bool((fixed == -12345).all()) if mode == 2 else True
                    c = {'style': style, 'mode': mode, 'fixed_max_abs': fixederr, 'final_max_abs': finalerr, 'omission_correct': omitted}
                    assert omitted and (fixederr is None or fixederr == 0) and (finalerr is None or finalerr == 0), c
                    microchecks.append(c)
            write(w.out / 'micro_correctness.json', microchecks)
            if args.phase == 'micro_profile':
                fn = micros[args.style]
                fn(*views, cutlass.Int32(32), cutlass.Int32(args.mode), stream)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push('K3_RUNTIME_TARGET')
                fn(*views, cutlass.Int32(32), cutlass.Int32(args.mode), stream)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
            else:
                graphs = {}
                configs = [('RUNTIME', 1), ('RUNTIME', 2), ('RUNTIME', 3), ('COALESCED', 3)]
                for (style, mode) in configs:
                    for repeat in args.repeats:
                        rt = tuple(views) + (cutlass.Int32(repeat), cutlass.Int32(mode), stream)
                        graphs[style, mode, repeat] = graph_for(torch, micros[style], rt, {})
                rows = []
                rng = random.Random(8765 + args.trial)
                keys = list(graphs)
                rng.shuffle(keys)
                for key in keys:
                    (style, mode, repeat) = key
                    for _ in range(20):
                        graphs[key].replay()
                    torch.cuda.synchronize()
                    for sample in range(args.samples):
                        if sample % 25 == 0:
                            base.check_exclusive_gpu()
                        (begin, end) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        begin.record()
                        graphs[key].replay()
                        end.record()
                        end.synchronize()
                        rows.append({'trial': args.trial, 'style': style, 'mode': mode, 'repeat': repeat, 'sample': sample, 'us': begin.elapsed_time(end) * 1000, 'cycle_at_1845MHz': begin.elapsed_time(end) * 1845000})
                with (w.out / 'slope_samples.csv').open('x', newline='') as f:
                    cw = csv.DictWriter(f, fieldnames=list(rows[0]))
                    cw.writeheader()
                    cw.writerows(rows)
                write(w.out / 'slope_protocol.json', {'grid': 32, 'block': 256, 'rotating_slots': 8, 'logical_bytes_each_CTA_repeat': 32768, 'GMEM_bytes_each_CTA_repeat': {'mode1': 32768, 'mode2': 32768, 'mode3': 65536}, 'working_set_output_bytes': fixed.numel() * 4 + final.numel() * 4, 'R': args.repeats, 'input_before_loop': 'GMEM->TMEM once', 'no_MMA': True, 'no_TMA': True, 'Event_endpoint': 'kernel complete, not physical HBM flush'})
        write(w.out / 'manifest.json', {'source_commit': w.helper.SOURCE_COMMIT, 'generated_sha256': base.sha(args.implementation), 'runner_sha256': base.sha(Path(__file__)), 'helper_micro_sha256': base.sha(ROOT / 'final_helper.py'), 'case': args.case, 'trial': args.trial, 'gpu_before': w.before, 'gpu_after': base.gpu_snapshot(), 'argv': sys.argv, 'shared_runtime_mode_binary': True, 'binary': {k: w.helper.inventory(r['binary_dir']) for (k, r) in w.compiled.items()}})
        base.check_exclusive_gpu()
        print(json.dumps({'phase': args.phase, 'status': 'PASS', 'out': str(w.out)}), flush=True)
    finally:
        w.close()
if __name__ == '__main__':
    main()
