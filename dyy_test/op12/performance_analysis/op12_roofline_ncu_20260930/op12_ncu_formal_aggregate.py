#!/usr/bin/env python3
"""Validate 15 paired CP4 NCU reports and build the machine-readable handoff."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


SHAPES = {8192: 1024, 16384: 2048, 32768: 4096}
STAGES = ('CPDeltaRuleTPrecomputeSm100', 'CPDeltaRuleMNPrecomputeUtcmma1Sm100',
          'CPDeltaRuleFixupUtcmmaSm100', 'CPDeltaRulePrefillTcgen05Sm100')
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum')


def parse_csv(path):
    with path.open(newline='') as handle:
        table = list(csv.reader(handle))
    if len(table) != 6:
        raise AssertionError(f'{path}: expected header, units and exactly four Ki rows')
    header, units = table[:2]
    named = dict(zip(header, range(len(header))))
    if len(named) != len(header):
        raise AssertionError(f'{path}: duplicate CSV headers')
    for name, unit in zip(METRICS, ('byte', 'byte', 'ns')):
        if units[named[name]] != unit:
            raise AssertionError(f'{path}: {name} unit {units[named[name]]}, expected {unit}')
    rows = [dict(zip(header, values)) for values in table[2:]]
    context_stream = set()
    result = []
    for index, row in enumerate(rows):
        if row['ID'] != str(index) or STAGES[index] not in row['Kernel Name']:
            raise AssertionError(f'{path}: Ki order/name mismatch at {index}')
        if 'OP12_CP4_COLD' not in row['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg']:
            raise AssertionError(f'{path}: Ki missing measured NVTX range')
        context_stream.add((row['Context'], row['Stream']))
        result.append({'key': f'K{index+1}',
                       'dram_read_bytes': int(row[METRICS[0]]),
                       'dram_write_bytes': int(row[METRICS[1]]),
                       'duration_ns': int(row[METRICS[2]]),
                       'replayer_passes': int(row['profiler__replayer_passes'])})
    if len(context_stream) != 1 or len({r['replayer_passes'] for r in result}) != 1:
        raise AssertionError(f'{path}: context/stream/replay count mismatch')
    return result


def summary(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values),
            'max': max(values), 'population_sd': statistics.pstdev(values)}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    samples = []
    for shape, chunk in SHAPES.items():
        for sample_no in range(1, 6):
            path = root / 'formal' / f'S{shape}' / 'evict4' / f'sample{sample_no:02d}_idle0ms'
            complete = json.loads((path / 'complete.json').read_text())
            command = json.loads((path / 'command.json').read_text())
            rows = parse_csv(path / 'metrics_raw.csv')
            if rows[0]['replayer_passes'] != 1:
                raise AssertionError(f'{path}: core metrics unexpectedly require multiple passes')
            if sha256(path / 'cp4.ncu-rep') != complete['report_sha256'] or \
               sha256(path / 'metrics_raw.csv') != complete['csv_sha256']:
                raise AssertionError(f'{path}: report/CSV hash mismatch')
            passes = sorted((path / 'app_passes').glob('app_pass_*/pass_manifest.json'))
            if complete['app_pass_count'] != len(passes) or len(passes) != rows[0]['replayer_passes']:
                raise AssertionError(f'{path}: app replay pass count mismatch')
            manifests = [json.loads(p.read_text()) for p in passes]
            if any(m['shape'] != shape or m['chunk'] != chunk or m['sample'] != sample_no or
                   m['evict_factor'] != 4 or m['tail_mode'] != 'none' or
                   m['graph']['kernel_count'] != 4 or not all(m['checks'].values())
                   for m in manifests):
                raise AssertionError(f'{path}: CP/shape/eviction/correctness validation failed')
            hashes = {json.dumps(m['output_hashes'], sort_keys=True) for m in manifests}
            if len(hashes) != 1:
                raise AssertionError(f'{path}: output changed across application replay passes')
            argv = command['argv']
            for term in ('--replay-mode', 'application', '--cache-control', 'none',
                         '--clock-control', 'none', '--app-replay-mode', 'strict',
                         '--app-replay-match', 'all', '--graph-profiling', 'node'):
                if term not in argv:
                    raise AssertionError(f'{path}: missing effective command term {term}')
            before = manifests[0]['telemetry_before'].split(',')
            after = manifests[-1]['telemetry_after'].split(',')
            rep = path / 'cp4.ncu-rep'
            for row in rows:
                row['rep_path'] = str(rep)
                row['raw_csv_path'] = str(path / 'metrics_raw.csv')
                row['sm_clock_mhz'] = int(before[0].strip())
                row['sm_clock_after_mhz'] = int(after[0].strip())
            samples.append({'S': shape, 'C': chunk, 'run_id': sample_no,
                'sample_tag': f'S{shape}_{sample_no:02d}',
                'output_hash': manifests[0]['output_hashes']['output_sha256'],
                'final_state_hash': manifests[0]['output_hashes']['final_state_sha256'],
                'correct': True, 'kernels': rows,
                'app_pass_count': len(passes), 'frozen_input_sha256': manifests[0]['frozen_input_sha256'],
                'harness_sha256': manifests[0]['harness_sha256'],
                'source_commit': manifests[0]['source_commit'],
                'l2_bytes': manifests[0]['l2_bytes'],
                'gpu_telemetry_before': manifests[0]['telemetry_before'],
                'gpu_telemetry_after': manifests[-1]['telemetry_after'],
                'command_json_path': str(path / 'command.json')})
    if len(samples) != 15:
        raise AssertionError('expected 15 complete formal samples')
    for shape in SHAPES:
        group = [r for r in samples if r['S'] == shape]
        if len({r['output_hash'] for r in group}) != 1 or len({r['final_state_hash'] for r in group}) != 1:
            raise AssertionError(f'S{shape}: output differs among five independent samples')
    stats = []
    for shape in SHAPES:
        for stage in range(4):
            group = [r['kernels'][stage] for r in samples if r['S'] == shape]
            stats.append({'S': shape, 'key': f'K{stage+1}',
                'read_bytes': summary([r['dram_read_bytes'] for r in group]),
                'write_bytes': summary([r['dram_write_bytes'] for r in group]),
                'total_bytes': summary([r['dram_read_bytes'] + r['dram_write_bytes'] for r in group]),
                'duration_ns': summary([r['duration_ns'] for r in group])})
    first_command = json.loads((root / 'formal/S8192/evict4/sample01_idle0ms/command.json').read_text())
    out = {'capture': {'evidence_type': 'formal',
            'ncu_version': first_command['ncu_version'],
            'gpu_preflight': first_command['gpu_preflight'],
            'replay_mode': 'application', 'app_replay_mode': 'strict',
            'app_replay_match': 'all', 'cache_control': 'none', 'clock_control': 'none',
            'graph_profiling': 'node', 'evict_factor': 4,
            'l2_bytes': samples[0]['l2_bytes'],
            'sm_clock_mhz_meaning': 'nvidia-smi snapshot before eviction, not per-kernel frequency',
            'metric_units': {'dram_read_bytes': 'byte', 'dram_write_bytes': 'byte',
                             'duration_ns': 'ns'},
            'tail_writeback': 'separate diagnostic; not allocated to Ki'},
           'samples': samples, 'statistics': stats}
    path = root / 'OP12_CP4_FORMAL_NCU_SAMPLES.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'file': str(path), 'samples': len(samples), 'kernel_records': 60}))


if __name__ == '__main__':
    main()
