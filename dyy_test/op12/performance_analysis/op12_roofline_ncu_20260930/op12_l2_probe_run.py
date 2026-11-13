#!/usr/bin/env python3
"""Profile isolated L2 probe with per-application replay and archive evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/op12_roofline_ncu_20260930')
NCU = Path('/opt/nvidia/nsight-compute/2026.2.1/ncu')
PY = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python')
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum',
           'lts__t_sectors_op_read_lookup_hit.sum',
           'lts__t_sectors_op_read_lookup_miss.sum', 'gpu__time_duration.sum')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, choices=(2, 4, 8), required=True)
    args = parser.parse_args()
    occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    if occupancy and occupancy != 'No running processes found':
        raise RuntimeError('GPU occupied: ' + occupancy)
    dest = ROOT / 'l2_probe' / f'evict{args.factor}'
    dest.mkdir(parents=True, exist_ok=False)
    command = [str(NCU), '--config-file', 'off', '--target-processes', 'application-only',
               '--nvtx', '--nvtx-include', 'OP12_L2_PROBE/',
               '--replay-mode', 'application', '--app-replay-mode', 'strict',
               '--app-replay-match', 'all', '--cache-control', 'none',
               '--clock-control', 'none', '--metrics', ','.join(METRICS),
               '--disable-extra-suffixes', '--apply-rules', 'off',
               '--export', str(dest / 'probe'), str(PY), str(ROOT / 'op12_l2_probe.py'),
               '--factor', str(args.factor), '--out', str(dest / 'passes')]
    (dest / 'command.json').write_text(json.dumps({'argv': command}, indent=2) + '\n')
    with (dest / 'ncu_stdout.log').open('x') as stdout, (dest / 'ncu_stderr.log').open('x') as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=900)
    if proc.returncode:
        raise RuntimeError(f'NCU exit {proc.returncode}: {dest / "ncu_stderr.log"}')
    report = dest / 'probe.ncu-rep'
    imported = subprocess.run([str(NCU), '--config-file', 'off', '--import', str(report),
                               '--csv', '--page', 'raw', '--print-units', 'base'],
                              capture_output=True, text=True, check=True, timeout=120)
    (dest / 'metrics_raw.csv').write_text(imported.stdout)
    (dest / 'import_stderr.log').write_text(imported.stderr)
    results = list((dest / 'passes').glob('pass_*/probe.json'))
    if not results:
        raise RuntimeError('No probe pass validation')
    print(json.dumps({'factor': args.factor, 'application_passes': len(results),
                      'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
                      'csv': str(dest / 'metrics_raw.csv')}), flush=True)


if __name__ == '__main__':
    main()
