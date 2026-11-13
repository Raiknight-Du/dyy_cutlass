#!/usr/bin/env python3

import argparse

import numpy as np


def summarize_axis(diff, axis, tile, name):
    counts = np.count_nonzero(diff, axis=axis)
    total = diff.shape[axis]
    print(f"{name} mismatch range: min={counts.min()} max={counts.max()}")
    groups = (counts.size + tile - 1) // tile
    for group in range(groups):
        begin = group * tile
        end = min(begin + tile, counts.size)
        mismatches = int(counts[begin:end].sum())
        elements = (end - begin) * total
        print(
            f"  {name} tile {group:4d} [{begin:6d},{end:6d}): "
            f"{mismatches:10d}/{elements:10d} "
            f"({100.0 * mismatches / elements:8.4f}%)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("packed")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m-tile", type=int, default=256)
    parser.add_argument("--n-tile", type=int, default=192)
    args = parser.parse_args()

    shape = (args.m, args.n)
    baseline = np.memmap(args.baseline, dtype=np.uint16, mode="r", shape=shape)
    packed = np.memmap(args.packed, dtype=np.uint16, mode="r", shape=shape)
    diff = baseline != packed

    mismatch_count = int(np.count_nonzero(diff))
    element_count = args.m * args.n
    print(
        f"mismatch: {mismatch_count}/{element_count} "
        f"({100.0 * mismatch_count / element_count:.6f}%)"
    )

    if mismatch_count:
        first = np.argwhere(diff)[:32]
        print("first mismatches (m, n, baseline_bits, packed_bits):")
        for m, n in first:
            print(
                f"  {int(m):6d} {int(n):4d} "
                f"0x{int(baseline[m, n]):04x} 0x{int(packed[m, n]):04x}"
            )

    print("\nBy N tile:")
    summarize_axis(diff, axis=0, tile=args.n_tile, name="N")
    print("\nBy M tile:")
    summarize_axis(diff, axis=1, tile=args.m_tile, name="M")

    print("\nM/N tile mismatch matrix (percentage):")
    m_tiles = (args.m + args.m_tile - 1) // args.m_tile
    n_tiles = (args.n + args.n_tile - 1) // args.n_tile
    for mt in range(m_tiles):
        m0 = mt * args.m_tile
        m1 = min(m0 + args.m_tile, args.m)
        values = []
        for nt in range(n_tiles):
            n0 = nt * args.n_tile
            n1 = min(n0 + args.n_tile, args.n)
            block = diff[m0:m1, n0:n1]
            values.append(100.0 * np.count_nonzero(block) / block.size)
        print(f"  M tile {mt:4d}: " + " ".join(f"{value:7.3f}" for value in values))


if __name__ == "__main__":
    main()
