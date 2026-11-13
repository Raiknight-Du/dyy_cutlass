"""Four bounded K1 correctness cases; execute only under the lead GPU guard.

This file has no CUDA imports at module scope. --dry-run and used_tiles are CPU
only. The actual children invoke the existing full-kernel validation harness.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = (
    dict(name="fp16_ragged", dtype="fp16", lens=[1, 63, 64, 65, 127, 257], kh=16, ho=16),
    dict(name="fp16_empty_gqa", dtype="fp16", lens=[0, 1, 0, 65], kh=4, ho=16),
    dict(name="bf16_ragged", dtype="bf16", lens=[1, 63, 64, 65, 127, 257], kh=16, ho=16),
    dict(name="bf16_empty_gqa", dtype="bf16", lens=[0, 1, 0, 65], kh=4, ho=16),
)


def used_tiles(lengths, blk=64):
    """Yield (workspace_slot, valid_length, absolute_token_start).

    Exact upstream chunk_bound/varlen_chunk_idx formula, including empty
    sequences. Workspace slots need not be densely packed.
    """
    start = 0
    for seq, length in enumerate(lengths):
        if length < 0:
            raise ValueError("sequence lengths must be nonnegative")
        m = min(seq, start)
        base = m + (start - m) // blk
        for chunk in range((length + blk - 1) // blk):
            yield base + chunk, min(blk, length - blk * chunk), start + blk * chunk
        start += length


def poison_used_tiles(output, lengths):
    """Call before a validation launch, on its stream (or synchronize after).

    Leave unused workspace holes zero, poison every element of used tiles.
    The harness's existing whole-output isfinite gate then proves full writes,
    including padding, without treating unused holes as failed outputs.
    """
    for slot, _, _ in used_tiles(lengths):
        output[slot].fill_(float("nan"))


def check_structural_output(output, lengths, beta, zero_k=False):
    """Independent exact structure checks; lazy torch import for GPU caller.

    PyTorch output axes are [tile, head, row, col]. The original physical
    gT[c,r] store transposes the lower inverse, giving an upper triangular T.
    """
    import torch
    for slot, valid, token_start in used_tiles(lengths):
        tile = output[slot]
        assert torch.isfinite(tile).all().item(), (slot, "missing/nonfinite writes")
        assert torch.count_nonzero(tile[:, valid:, :]).item() == 0, (slot, "padded rows")
        assert torch.count_nonzero(tile[:, :, valid:]).item() == 0, (slot, "padded columns")
        live = tile[:, :valid, :valid]
        assert torch.count_nonzero(torch.tril(live, diagonal=-1)).item() == 0, (slot, "lower triangle")
        expected_diag = (-beta[token_start:token_start + valid].T).to(tile.dtype)
        torch.testing.assert_close(torch.diagonal(live, dim1=-2, dim2=-1), expected_diag, rtol=0, atol=0)
        if zero_k:
            assert torch.count_nonzero(torch.triu(live, diagonal=1)).item() == 0, (slot, "zero-K offdiagonal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="validation_v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cases", default=",".join(c["name"] for c in CASES))
    args = parser.parse_args()
    selected = args.cases.split(",")
    unknown = set(selected) - {c["name"] for c in CASES}
    if unknown:
        parser.error("unknown cases: " + ",".join(sorted(unknown)))
    root = Path(__file__).resolve().parents[1]
    manifest = []
    for case in CASES:
        if case["name"] not in selected:
            continue
        slots = list(used_tiles(case["lens"]))
        bound_m = min(len(case["lens"]), sum(case["lens"]))
        workspace_slots = bound_m + (sum(case["lens"]) - bound_m) // 64
        assert len({x[0] for x in slots}) == len(slots)
        assert all(0 <= x[0] < workspace_slots for x in slots)
        tag = args.tag + "_" + case["name"]
        command = [sys.executable, str(root / "source" / "run_compare.py"),
                   "--mode", "validate", "--tag", tag,
                   "--s", str(sum(case["lens"])), "--dtype", case["dtype"],
                   "--lens", ",".join(map(str, case["lens"])),
                   "--kh", str(case["kh"]), "--ho", str(case["ho"]),
                   "--variants", "baseline,tcgen05,layout,both"]
        row = dict(case=case, command=command, used_tiles=slots,
                   workspace_slots=workspace_slots, tag=tag)
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not args.dry_run:
            assert os.environ.get("K1_GPU_SESSION"), "serialized lead GPU guard required"
            start = time.monotonic()
            completed = subprocess.run(command, check=False)
            row.update(returncode=completed.returncode, elapsed_seconds=time.monotonic() - start)
            (root / "results" / (args.tag + "_suite.json")).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            if completed.returncode:
                raise SystemExit(completed.returncode)
    print("DRY_RUN_COMPLETE" if args.dry_run else "VALIDATION_SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
