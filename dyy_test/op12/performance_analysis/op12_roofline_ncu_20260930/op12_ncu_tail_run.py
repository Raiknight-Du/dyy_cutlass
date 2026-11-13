#!/usr/bin/env python3
"""Isolated post-CP dirty-writeback diagnostic, never used as main sample."""

import argparse
import csv
import json
from pathlib import Path
import subprocess


ROOT = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/op12_roofline_ncu_20260930')
NCU = Path('/opt/nvidia/nsight-compute/2026.2.1/ncu')
PY = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python')
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=int, choices=(8192, 16384, 32768), required=True)
    parser.add_argument('--mode', choices=('after_cp', 'control'), required=True)
    args = parser.parse_args()
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    if occupied and occupied != 'No running processes found':
        raise RuntimeError('GPU occupied: ' + occupied)
    dest = ROOT / 'tail_diagnostic' / f'S{args.shape}' / args.mode
    dest.mkdir(parents=True, exist_ok=False)
    tag = 'OP12_CP4_WITH_TAIL/' if args.mode == 'after_cp' else 'OP12_CP4_TAIL_CONTROL/'
    argv = [str(NCU), '--config-file', 'off', '--target-processes', 'application-only',
        '--nvtx', '--nvtx-include', tag, '--replay-mode', 'application',
        '--app-replay-mode', 'strict', '--app-replay-match', 'all',
        '--cache-control', 'none', '--clock-control', 'none', '--graph-profiling', 'node',
        '--metrics', ','.join(METRICS), '--disable-extra-suffixes', '--apply-rules', 'off',
        '--export', str(dest / 'tail'), str(PY), str(ROOT / 'op12_ncu_cold.py'),
        '--shape', str(args.shape), '--sample', '900', '--evict-factor', '4',
        '--tail-mode', args.mode, '--out', str(dest / 'app_passes')]
    (dest / 'command.json').write_text(json.dumps({'argv': argv}, indent=2) + '\n')
    with (dest / 'ncu_stdout.log').open('x') as stdout, (dest / 'ncu_stderr.log').open('x') as stderr:
        proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=1200)
    if proc.returncode:
        raise RuntimeError(f'NCU exit {proc.returncode}: {dest / "ncu_stderr.log"}')
    report = dest / 'tail.ncu-rep'
    imported = subprocess.run([str(NCU), '--config-file', 'off', '--import', str(report),
                               '--csv', '--page', 'raw', '--print-units', 'base'],
                              capture_output=True, text=True, check=True, timeout=120)
    (dest / 'metrics_raw.csv').write_text(imported.stdout)
    with (dest / 'metrics_raw.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('ID', '').isdigit()]
    if len(rows) != (5 if args.mode == 'after_cp' else 1):
        raise RuntimeError(f'expected 5/1 kernels, got {len(rows)}')
    summary = {'shape': args.shape, 'mode': args.mode,
               'kernels': [{'id': int(r['ID']), 'name': r['Kernel Name'],
                            'read_bytes': int(r[METRICS[0]]),
                            'write_bytes': int(r[METRICS[1]]),
                            'duration_ns': int(r[METRICS[2]]),
                            'passes': int(r['profiler__replayer_passes'])} for r in rows],
               'app_passes': len(list((dest / 'app_passes').glob('app_pass_*/pass_manifest.json')))}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
