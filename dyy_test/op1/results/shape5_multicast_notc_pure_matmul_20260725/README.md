# Shape5 pure-matmul multicast / NO-TC experiment

Problem:

```text
M=8192, N=768, K=4096
MmaTile=256x192x256
ElementC=void
Epilogue=Identity(convert(accumulator))
Persistent deterministic random A/B/SFA/SFB input
```

Cluster controls:

| Config | Physical cluster | Logical VMNK | Extra multicast reuse |
|---|---|---|---|
| C2x1 | 2x1x1 | 2x1x1x1 | none beyond the 2SM atom |
| C2x2 | 2x2x1 | 2x1x2x1 | A/SFA across two N tiles |
| C4x1 | 4x1x1 | 2x2x1x1 | B/SFB across two logical M tiles |

Three binaries are built from the same pure-matmul `.cu`:

- `full`: normal block-scaled UMMA.
- `notc`: the isolated header overlay removes only `cute::gemm`; TMA,
  SFA/SFB UTCCP, pipeline waits/releases, TMEM/epilogue shell, and D store
  remain.
- `notma`: the isolated `include_notma` overlay removes A/B/SFA/SFB TMA
  and bypasses the mainloop full/empty barriers while retaining UTCCP,
  block-scaled UMMA, the accumulator pipeline, and the epilogue.

Event timing uses five independent samples with 50 timed iterations each.
NCU uses one profiled timed launch after one warmup launch, with cache and
clock controls disabled.

Generated analysis:

- `comparison_summary.csv`
- `ANALYSIS_CN.md`
