# op12 native CUDA Gated DeltaNet prefill

This directory contains two explicitly separated SM100 GDN prefill backends:

- independent CUDA C correctness/performance-development backends
  (`scalar` and 64-token `wmma`);
- an upstream-derived `tcgen05` parity backend pinned to the archived
  FlashInfer source.  This is the default B200 performance path and is not
  represented as an independent reimplementation.

The correctness path is a persistent recurrent kernel.  One CUDA block owns
one `(sequence, output-head)` state, keeps the 128x128 FP32 working state in
shared memory, and advances tokens in strict order.  A separate context-
parallel path is also implemented: partition-local affine `(M,N)` precompute,
cross-partition fixup, and independent final prefill kernels.  The public
wrapper covers the complete SM100 data contract (TND varlen, FP16/BF16 Q/K/V,
GQA/GVA, optional initial/final state, state-pool indices, and checkpoints).
Select a backend without changing the Python API:

```bash
OP12_NATIVE_BACKEND=tcgen05  # default, fixed upstream-derived CuTe DSL path
OP12_NATIVE_BACKEND=wmma     # independent 64-token CUDA WMMA/HMMA path
OP12_NATIVE_BACKEND=scalar   # independent strict-token reference path
```

On B200, the WMMA path emits `HMMA.16816.F32` but remains synchronization
limited.  The parity path emits SM100 `UTCHMMA`/TMEM/TMA instructions and its
40-case median latency ratio versus the fixed FlashInfer baseline is 1.017
(range 1.002-1.044).

The upstream fixed commit used for differential tests is
`e77a4a0d276367895c3b50a642fd8f326c03fb72`.
