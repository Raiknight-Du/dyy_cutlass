#!/usr/bin/env python3
"""Independent hot/cold L2 read probe, isolated from CP captures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out / f'pass_{os.getpid()}_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    props = torch.cuda.get_device_properties(0)
    target = torch.rand((8 * 1024 * 1024 // 4,), device='cuda', dtype=torch.float32)
    eviction = torch.rand((props.L2_cache_size * args.factor // 4,),
                          device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = target.sum()
    stream.synchronize()
    torch.cuda.nvtx.range_push('OP12_L2_PROBE')
    torch.cuda.nvtx.range_push('HOT_TARGET_READ')
    with torch.cuda.stream(stream):
        hot = target.sum()
    torch.cuda.nvtx.range_pop()
    stream.synchronize()
    torch.cuda.nvtx.range_push('EVICT_READ_ONLY')
    with torch.cuda.stream(stream):
        evict_value = eviction.sum()
    torch.cuda.nvtx.range_pop()
    stream.synchronize()
    torch.cuda.nvtx.range_push('COLD_TARGET_READ')
    with torch.cuda.stream(stream):
        cold = target.sum()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    stream.synchronize()
    result = {'factor': args.factor, 'l2_bytes': props.L2_cache_size,
              'target_bytes': target.numel() * target.element_size(),
              'evict_bytes': eviction.numel() * eviction.element_size(),
              'warm': warm.item(), 'hot': hot.item(), 'cold': cold.item(),
              'eviction_checksum': evict_value.item()}
    assert warm.item() == hot.item() == cold.item()
    (out / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'probe_pass_complete', **result}), flush=True)


if __name__ == '__main__':
    main()
