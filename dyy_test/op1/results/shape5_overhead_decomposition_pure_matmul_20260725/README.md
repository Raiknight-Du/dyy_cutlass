# Shape5 pure-matmul overhead decomposition

This directory contains the reproducible shape5 overhead experiment.

Key files:

- `op1_shape5_overhead_pure_matmul.cu`: pure-matmul harness, config36 only
  when built with `OVERHEAD_CONFIG36_ONLY`.
- `empty_cluster_baseline.cu`: 0-SMEM, 227.328-KiB-SMEM, and cluster-sync
  fixed-cost probes.
- `include_shell_*`: isolated CUTLASS header overlays for each ablation.
- `run_experiment.py`: build, Event timing, and NCU collection.
- `analyze_results.py`: summary and SourceCounters aggregation.
- `event_raw.csv`, `event_summary.csv`: five by 50 Event timing.
- `ncu/`: reports, raw/details CSV, and SourceCounters output.
- `decomposition_summary.csv`: primary latency decomposition table.
- `empty_cluster_summary.csv`: fixed cluster/resource baseline.
- `ncu_key_metrics.csv`: instruction and scheduler validation.
- `source_hotspots.csv`: CUDA/SASS hotspot aggregation.
- `source_reference/`: exact CUTLASS headers used for source analysis.
- `ANALYSIS_CN.md`: Chinese analysis report.

The `shell_nodstore` and `shell_noepi` epilogue overlays include a local
`collective_epilogue.hpp` so CUTLASS's same-directory quoted includes resolve
to the modified epilogue header. NCU verifies that their TMA-store and TMEM-load
instruction counts are zero, respectively.
