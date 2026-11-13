#!/usr/bin/env python3
"""Extract paired per-kernel values from CP4 NCU raw CSV reports."""

import csv
import json
from pathlib import Path
import re


ROOT = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/op12_roofline_ncu_20260930')
STAGES = ('CPDeltaRuleTPrecomputeSm100', 'CPDeltaRuleMNPrecomputeUtcmma1Sm100',
          'CPDeltaRuleFixupUtcmmaSm100', 'CPDeltaRulePrefillTcgen05Sm100')
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum')


def parse_capture(path):
    with path.open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('ID', '').isdigit()]
    if len(rows) != 4:
        raise AssertionError(f'{path}: expected 4 kernel records, found {len(rows)}')
    output = []
    for stage, row in enumerate(rows):
        if int(row['ID']) != stage or STAGES[stage] not in row['Kernel Name']:
            raise AssertionError(f'{path}: kernel stage order mismatch at {stage}')
        if 'OP12_CP4_COLD' not in row['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg']:
            raise AssertionError(f'{path}: missing measured NVTX range')
        if int(row['profiler__replayer_passes']) != 1:
            raise AssertionError(f'{path}: expected one application replay pass per kernel')
        data = {'kernel': f'K{stage+1}', 'name': row['Kernel Name'],
                'context': row['Context'], 'stream': row['Stream'],
                'read_bytes': int(row[METRICS[0]]),
                'write_bytes': int(row[METRICS[1]]),
                'duration_ns': int(row[METRICS[2]])}
        data['total_bytes'] = data['read_bytes'] + data['write_bytes']
        output.append(data)
    if len({r['stream'] for r in output}) != 1 or len({r['context'] for r in output}) != 1:
        raise AssertionError(f'{path}: stages not in one context and stream')
    return output


def main():
    records = []
    for capture in sorted((ROOT / 'pilot').glob('S*/evict*/sample*/metrics_raw.csv')):
        bits = capture.relative_to(ROOT / 'pilot').parts
        # Idle/writeback controls are diagnostics, not members of the 2/4/8x convergence trio.
        if not re.fullmatch(r'sample\d\d', bits[2]):
            continue
        shape = int(bits[0][1:]); factor = int(bits[1][5:]); sample = int(bits[2][6:])
        for entry in parse_capture(capture):
            records.append({'shape': shape, 'evict_factor': factor, 'sample': sample,
                            **{k:v for k,v in entry.items() if k != 'name'},
                            'source_csv': str(capture)})
    by_key = {(r['shape'], r['evict_factor'], r['kernel']): r for r in records}
    comparison = []
    for shape in (8192, 16384, 32768):
        for kernel in ('K1', 'K2', 'K3', 'K4'):
            row = {'shape': shape, 'kernel': kernel}
            for factor in (2, 4, 8):
                r = by_key[(shape, factor, kernel)]
                for key in ('read_bytes', 'write_bytes', 'total_bytes', 'duration_ns'):
                    row[f'{key}_{factor}x'] = r[key]
            for key in ('read_bytes', 'write_bytes', 'total_bytes'):
                a, b = row[f'{key}_4x'], row[f'{key}_8x']
                row[f'{key}_4to8_abs'] = b-a
                row[f'{key}_4to8_pct'] = None if max(abs(a), abs(b)) < 1_000_000 else 100*(b-a)/max(abs(a),abs(b))
            comparison.append(row)
    (ROOT / 'pilot_summary.json').write_text(json.dumps({'rows': records, 'convergence': comparison}, indent=2) + '\n')
    with (ROOT / 'pilot_convergence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison[0].keys())
        writer.writeheader(); writer.writerows(comparison)
    print(json.dumps({'captures': len(records)//4, 'kernel_rows': len(records),
                      'convergence': comparison}, indent=2))


if __name__ == '__main__':
    main()
