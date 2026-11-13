# op12 native CUDA GDN baseline results

## Scope

- GPU: NVIDIA B200, compute capability 10.0, 148 SM.
- CUDA toolkit: 13.0; PyTorch: 2.11.0+cu130.
- Differential oracle: FlashInfer commit
  `e77a4a0d276367895c3b50a642fd8f326c03fb72`.
- Implementation: independent CUDA C recurrent kernel; no FlashInfer kernel code
  is linked or modified.

## Correctness milestone

`pytest tests/test_correctness.py` passes 39/39 cases.  Coverage includes:

- FP16 and BF16 Q/K/V;
- MHA, GQA, and GVA head mapping;
- packed TND variable-length input, full/tail chunks, and zero-length sequences;
- optional initial state and final state;
- FP32, FP16, BF16, FP8 E4M3, and FP8 E5M2 state storage;
- indexed state pools and in-place pool update;
- state checkpoints every 64 tokens;
- a real three-kernel context-parallel path (M/N affine precompute, fixup,
  final prefill), compared with upstream `use_cp=True` for FP16/BF16,
  MHA/GQA/GVA, and varlen inputs.

The machine-readable result is `pytest_correctness.xml`.

## Equal-method performance matrix

The full run contains 40 configurations x 3 independent processes, with 20
warmups and 100 CUDA-event samples per configuration.  All 120 case results
passed and were finite.  The maximum cross-process CV of the median is 0.0154%.
Common conditions are initial/final state enabled, Q/V normal, K L2-normalized
outside the timed region, FP32 gate/beta, and scale=1/sqrt(128).

| FP16 B1/H16 shape | native non-CP | native CP | FlashInfer non-CP | FlashInfer CP |
|---|---:|---:|---:|---:|
| S=8192 | 47.5317 ms | 13.7508 ms | 0.253696 ms | 0.144944 ms |
| S=16384 | 94.4073 ms | 25.2443 ms | 0.483072 ms | 0.215648 ms |
| S=32768 | 188.7442 ms | 48.2640 ms | 0.941888 ms | 0.352016 ms |

Native CP is 3.46x, 3.74x and 3.91x faster than native non-CP at the three
lengths, but remains 94.9x, 117.1x and 137.1x slower than pinned FlashInfer CP.
Native non-CP is 187.4x to 200.4x slower than pinned FlashInfer non-CP.  This is
a feature-equivalent CUDA C baseline, not a tensor-core performance match.

BF16 and FP16 are effectively identical on the native path because both are
converted to FP32 and execute the same scalar FMA core.  Machine-readable
evidence is in `benchmark_native_summary.csv` and
`benchmark_native_vs_flashinfer.csv`.

## NCU evidence and bottleneck

Profiles are stored on B200 under `validation/ncu/*.ncu-rep`; local CSV/SASS
exports and SHA256 manifests are retained alongside this report.

| metric | value |
|---|---:|
| grid / block | 16 CTA / 256 threads |
| duration (B1/H16/S32768 FP16) | 188.06 ms |
| registers | 70/thread |
| dynamic shared memory | 68.10 KiB/CTA |
| waves/SM | 0.04 |
| achieved occupancy on active SM | 12.50% |
| compute throughput | 4.34% |
| DRAM throughput | 0.03% |
| L1/TEX throughput | 41.51% |
| eligible warps/scheduler | 0.34 |
| no-eligible cycles | 70.15% |
| tensor-core instructions | none observed |
| local/shared spilling | 0 |

The dominant causes are structural: only BxHo=16 independent blocks, scalar
FP32 FMA instead of tcgen05 tensor-core GEMM, strict token recurrence, and
barrier-heavy shared-state updates.  NCU attributes about 33.9% of the average
issue interval to CTA barrier stalls.  The optimized backend must move to the
64-token block formulation and context-parallel partitioning; tuning this
scalar recurrence cannot close the gap.

For B8/H16/S4096 (the same total 32768 tokens), grid increases from 16 to 128,
NCU duration falls from 188.06 ms to 23.64 ms and full-GPU SM throughput rises
from 4.34% to 34.56%.  Active-CTA behavior is otherwise unchanged: 0 tensor
activity, 0.34 eligible warps/scheduler and about 70.16% no-eligible cycles.

The native CP path always creates eight partitions/head for the three main
lengths, so affine and final grids are `(8,16,1)=128 CTA` and fixup remains
`(1,16,1)=16 CTA`.  FP16 stage durations are:

| S | affine M/N | fixup | final | NCU share |
|---:|---:|---:|---:|---|
| 8192 | 5.81 ms | 2.55 ms | 5.74 ms | 41.2% / 18.1% / 40.7% |
| 16384 | 11.70 ms | 2.55 ms | 11.38 ms | 45.6% / 9.9% / 44.4% |
| 32768 | 23.38 ms | 2.53 ms | 22.75 ms | 48.0% / 5.2% / 46.8% |

Affine and final scale with tokens.  Fixup is nearly constant because it scans
the same eight partitions/head, but it has only 16 CTAs, 2.46-2.48% SM
throughput and about 85.2% no-eligible cycles.  SASS contains FFMA/LDS/STS/BAR
and no HMMA/UTCHMMA/tcgen05 instruction; Tensor pipe activity is 0% in every
native stage.

Compute Sanitizer memcheck reports 0 errors for representative non-CP and CP
paths (`compute_sanitizer_noncp.log`, `compute_sanitizer_cp.log`).

## Next implementation gate

1. Implement the seven 64-token logical GEMMs with CUTLASS/tcgen05 FP16/BF16
   MMA and FP32 accumulation.
2. Preserve the existing Python contract and differential suite unchanged.
3. Replace the scalar CP affine/final cores with the same 64-token block
   algebra; keep the prefix fixup in FP32 and parallelize/scan it when partition
   count grows.
4. Reuse the unchanged 40-shape/3-trial and NCU harness as the optimization
   acceptance gate.
