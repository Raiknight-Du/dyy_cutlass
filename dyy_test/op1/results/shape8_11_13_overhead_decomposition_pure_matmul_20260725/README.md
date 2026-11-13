# Shape8/11/13 pure-matmul overhead decomposition

This experiment extends the verified shape5 ablation method to the best
pure-matmul configurations of shape8, shape11, and shape13.

Configurations:

- shape8: `M256_N256_K256_E64x64_C2x1x1`, config 0
- shape11: `M128_N192_K128_E128x64_C1x1x1`, config 78
- shape13: `M128_N256_K128_E128x128_C1x1x1`, config 63

Variants are cumulative where noted:

- `full`: complete pure matmul
- `notc`: remove UMMA only
- `notma`: remove A/B/SFA/SFB input TMA and matching barriers
- `shell_utccp`: remove input TMA and UMMA, retain UTCCP and epilogue
- `shell_noutccp`: additionally remove SFA/SFB SMEM-to-TMEM UTCCP
- `shell_nodstore`: additionally remove D SMEM-to-GMEM TMA store
- `shell_noepi`: additionally remove TMEM load, conversion, and SMEM staging

All GEMM timings use persistent random datasets, one in-process warmup, 25
independent samples, and 10 iterations per sample. NCU uses L2-hot,
non-clock-controlled profiling to match the project performance standard.

Remote path:

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/
  dyy_test/op1/results/
  shape8_11_13_overhead_decomposition_pure_matmul_20260725/
```

Reproduce:

```bash
python3 run_experiment.py
```
