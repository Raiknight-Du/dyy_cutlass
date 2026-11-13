# Shape 5/8/11/13 bottleneck probes

This directory contains a focused pure-matmul experiment. It does not modify
`dyy_test/op1/op1_best.cu`.

The matrix varies N tile size to trade block-level parallelism against repeated
per-CTA pipeline work. It also tests K512 where useful to reduce K-loop rounds.
All Event measurements follow the project standard: one in-process warmup,
10 timed iterations per sample, 25 independent samples, persistent random
A/B/SFA/SFB inputs, and an idle GPU.

Remote result directory:

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/
  dyy_test/op1/results/bottleneck_shape5_8_11_13_20260724/
```

Reproduce:

```bash
python3 run_experiment.py
python3 summarize.py
```
