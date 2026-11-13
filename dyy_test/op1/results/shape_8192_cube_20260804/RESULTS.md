# OP1 8192x8192x8192 performance

## Configuration

- Semantic: pure matmul, `ElementC=void`
- Problem: `M=N=K=8192`
- Config: `M256_N256_K256_E64x64_C2x1x1`
- Input: persistent random A/B/SFA/SFB dataset
- Timing: CUDA Event, 1 warmup, 10 iterations per sample
- Independent samples: 10

## Result

| Metric | Value |
|---|---:|
| Mean | 0.161054400 ms |
| Standard deviation | 0.000278580 ms |
| 95% CI half-width | 0.000172664 ms |
| Minimum | 0.160726000 ms |
| Maximum | 0.161549000 ms |
| Effective throughput | 6.826958 PFLOP/s |
| Utilization vs. 9 PFLOP/s | 75.855% |

The effective throughput uses `2MNK / time`.

## Reproduction

```bash
cd /home/ubuntu/workspace/dyy/dyy_cutlass
dyy_test/op1/results/shape_8192_cube_20260804/run_shape8192_standard.sh
```
