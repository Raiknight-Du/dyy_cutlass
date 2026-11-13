# Packed physical input experiment

This folder contains the reproducible source, raw benchmark data, Nsight
Compute evidence, and analysis for the op1 pure-matmul experiment that packs:

```text
A physical SMEM bytes + SFA physical SMEM bytes
B physical SMEM bytes + SFB physical SMEM bytes
```

Each K stage therefore issues two no-swizzle TMA loads instead of four
independent A/B/SFA/SFB loads. The UMMA, UTCCP, TMEM accumulator, pure-matmul
epilogue, D TMA store, tile, cluster, and stage policies are unchanged.

The stable remote source `dyy_test/op1/op1_best.cu` is not modified.

## Files

- `op1_packer.cu`: preprocessing library that materializes the final physical
  SMEM byte order in GMEM.
- `sm100_blockscaled_mma_warpspecialized_packed.hpp`: two-TMA packed mainloop.
- `op1_packed.cu`: common baseline/packed 13-shape harness.
- `tma_2sm_packed_probe.cu`: 2SM packed-descriptor mapping probe.
- `verify_13shape.py`: full-output bitwise digest validation.
- `run_13shape.py`: ten paired repeats, ten kernel iterations per sample.
- `event_raw.csv`: all 260 timed samples.
- `event_summary.csv`: per-shape means and standard deviations.
- `paired_summary.csv`: paired delta and 95% confidence interval.
- `ANALYSIS_CN.md`: implementation, correctness, performance, and NCU summary.

## Measurement scope

Input preprocessing is deliberately outside the timed region. These results
measure a packed-input kernel ABI, suitable when packing is performed offline,
amortized across reuse, or emitted directly by an upstream operator. They are
not an end-to-end claim for repacking fresh A and B before every GEMM.
