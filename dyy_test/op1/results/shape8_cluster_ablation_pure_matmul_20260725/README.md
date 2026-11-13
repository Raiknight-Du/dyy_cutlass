# Shape8 cluster ablation

Pure-matmul comparison at fixed:

```text
M=8192, N=4096, K=2048
Tile=256x256x256
Epilogue=64x64
Stage=Auto
```

Cluster variants:

- config 0: `2x1x1`
- config 22: `2x2x1`
- config 20: `4x1x1`

Each cluster is measured with Full, NO TC, NO TMA, and the same cumulative
shell decomposition used by the final shape5 analysis.

Timing follows the project standard: persistent random data, one in-process
warmup, 25 independent samples, and 10 iterations per sample.

Results and analysis:

- `ANALYSIS_CN.md`: Chinese bottleneck and cluster-ablation report
- `cluster_comparison.csv`: Event decomposition plus core NCU metrics
- `ncu_key_metrics.csv`: flattened metrics for all 21 NCU runs
- `empty_cluster_summary.csv`: empty/SMEM/sync cluster controls
- `analyze_results.py`: reproducible result summarizer
- `MMA_OVERLAP_ANALYSIS_CN.md`: paired NO-TMA experiment identifying which
  UTCCP/epilogue/D-store paths hide shape8 UMMA service time
- `run_mma_overlap.py`: builds the MMA-on counterparts and collects Event/NCU
- `run_mma_overlap_paired_event.py`: interleaved 25-sample paired timing
