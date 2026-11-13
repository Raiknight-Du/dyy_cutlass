#!/usr/bin/env python3
"""Run one serial, auditable CP4 NCU capture after checking GPU occupancy."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/op12_roofline_ncu_20260930')
NCU = Path('/opt/nvidia/nsight-compute/2026.2.1/ncu')
PY = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python')
HARNESS = ROOT / 'op12_ncu_cold.py'
METRICS = ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum']
STAGES = ('CPDeltaRuleTPrecomputeSm100', 'CPDeltaRuleMNPrecomputeUtcmma1Sm100',
          'CPDeltaRuleFixupUtcmmaSm100', 'CPDeltaRulePrefillTcgen05Sm100')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_csv(path):
    with path.open(newline='') as handle:
        table = list(csv.reader(handle))
    if len(table) != 6:
        raise RuntimeError(f'NCU CSV expected header, units, four records; got {len(table)} lines')
    header, units = table[:2]
    if len(set(header)) != len(header):
        raise RuntimeError('NCU CSV has duplicate column labels')
    names = dict(zip(header, range(len(header))))
    for metric, expected in zip(METRICS, ('byte', 'byte', 'ns')):
        if metric not in names or units[names[metric]] != expected:
            raise RuntimeError(f'NCU metric/unit mismatch for {metric}')
    result = []
    for index, values in enumerate(table[2:]):
        row = dict(zip(header, values))
        if len(values) != len(header) or int(row['ID']) != index or STAGES[index] not in row['Kernel Name']:
            raise RuntimeError(f'NCU record {index} is not ordered K{index+1}')
        if 'OP12_CP4_COLD' not in row['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg']:
            raise RuntimeError(f'NCU record {index} outside target NVTX range')
        if int(row['profiler__replayer_passes']) < 1:
            raise RuntimeError('invalid replayer pass count')
        numbers = [int(row[name]) for name in METRICS]
        if min(numbers) < 0 or numbers[2] == 0:
            raise RuntimeError('invalid HBM or time counter')
        result.append({'kernel': f'K{index+1}', 'name': row['Kernel Name'],
                       'context': row['Context'], 'stream': row['Stream'],
                       'passes': int(row['profiler__replayer_passes']),
                       'read_bytes': numbers[0], 'write_bytes': numbers[1],
                       'duration_ns': numbers[2]})
    if len({(r['context'], r['stream']) for r in result}) != 1:
        raise RuntimeError('NCU launch context/stream differs across K1-K4')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pilot', 'formal'), required=True)
    parser.add_argument('--shape', type=int, choices=(8192, 16384, 32768), required=True)
    parser.add_argument('--sample', type=int, required=True)
    parser.add_argument('--evict-factor', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--post-evict-idle-ms', type=int, default=0)
    args = parser.parse_args()
    occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    if occupancy and occupancy != 'No running processes found':
        raise RuntimeError('GPU occupied; preserving other work: ' + occupancy)
    dest = ROOT / args.mode / f'S{args.shape}' / f'evict{args.evict_factor}' / f'sample{args.sample:02d}_idle{args.post_evict_idle_ms}ms'
    dest.mkdir(parents=True, exist_ok=False)
    gpu_preflight = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip()
    command = [str(NCU), '--config-file', 'off', '--target-processes', 'application-only',
        '--nvtx', '--nvtx-include', 'OP12_CP4_COLD/', '--replay-mode', 'application',
        '--app-replay-mode', 'strict', '--app-replay-match', 'all',
        '--cache-control', 'none', '--clock-control', 'none', '--graph-profiling', 'node',
        '--metrics', ','.join(METRICS), '--disable-extra-suffixes', '--apply-rules', 'off',
        '--export', str(dest / 'cp4'), str(PY), str(HARNESS), '--shape', str(args.shape),
        '--sample', str(args.sample), '--evict-factor', str(args.evict_factor),
        '--post-evict-idle-ms', str(args.post_evict_idle_ms),
        '--out', str(dest / 'app_passes')]
    save(dest / 'command.json', {'argv': command, 'ncu_version': subprocess.check_output(
        [str(NCU), '--version'], text=True).strip(), 'metrics': METRICS,
        'harness_sha256': sha(HARNESS), 'start_wall_ns': time.time_ns(),
        'gpu_idle_preflight': True, 'gpu_preflight': gpu_preflight,
        'gpu_occupancy_query_result': occupancy})
    start = time.time()
    with (dest / 'ncu_stdout.log').open('x') as stdout, (dest / 'ncu_stderr.log').open('x') as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f'NCU exit {proc.returncode}; see {dest / "ncu_stderr.log"}')
    report = dest / 'cp4.ncu-rep'
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError('NCU did not produce .ncu-rep')
    imported = subprocess.run([str(NCU), '--config-file', 'off', '--import', str(report),
        '--csv', '--page', 'raw', '--print-units', 'base'], capture_output=True,
        text=True, timeout=120, check=True)
    (dest / 'metrics_raw.csv').write_text(imported.stdout)
    (dest / 'import_stderr.log').write_text(imported.stderr)
    passes = sorted((dest / 'app_passes').glob('app_pass_*/pass_manifest.json'))
    if not passes:
        raise RuntimeError('no successful application pass manifest')
    manifests = [json.loads(path.read_text()) for path in passes]
    if any(m['shape'] != args.shape or m['graph']['kernel_count'] != 4 or
           not all(m['checks'].values()) for m in manifests):
        raise RuntimeError('application replay pass validation failed')
    kernels = check_csv(dest / 'metrics_raw.csv')
    required_passes = max(row['passes'] for row in kernels)
    if len(passes) != required_passes or len({row['passes'] for row in kernels}) != 1:
        raise RuntimeError('application replay pass count does not match NCU report')
    if len({json.dumps(m['output_hashes'], sort_keys=True) for m in manifests}) != 1:
        raise RuntimeError('application replay output hash changed across passes')
    summary = {'mode': args.mode, 'shape': args.shape, 'sample': args.sample,
        'evict_factor': args.evict_factor, 'elapsed_s': time.time()-start,
        'app_pass_count': len(passes), 'app_passes': [str(p) for p in passes],
        'all_passes_checked': True, 'report_sha256': sha(report),
        'csv_sha256': sha(dest / 'metrics_raw.csv'), 'kernels': kernels}
    save(dest / 'complete.json', summary)
    print(json.dumps({'event': 'ncu_capture_complete', **summary}), flush=True)


if __name__ == '__main__':
    main()
