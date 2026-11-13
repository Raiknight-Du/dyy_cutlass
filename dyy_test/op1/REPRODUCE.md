# OP1 reproduction

[Implementation and timing notes](README.md) · [Official configuration table](results/op1_best_configs.csv) · [Standard retest](results/standard_retest_20260724/STANDARD_RETEST_REPORT_CN.md) · [Analysis and figures](analysis/)

Run the commands from the CUTLASS repository root on a B200 with an
SM100a-capable CUDA toolkit and host compiler. The CUTLASS headers used for
the recorded experiments are in the parent repository. Historical batch scripts
use the original B200 checkout path and may expect experiment-specific fixture
directories or binaries. Use a separate checkout or output directory when
rerunning a script that writes to the original results paths.

Full raw traces, profiler captures and historical run artifacts are in the
original B200 archive under
`/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/` and in the corresponding
2026-11-10 local experiment archives. Use the original fixtures for a
byte-identical rerun. The examples below run the same implementations with
the specified or regenerated input.

```bash
mkdir -p dyy_test/op1/bin
for variant in baseline best; do
  /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_100a \
    --expt-relaxed-constexpr -diag-suppress=20012 \
    -Iinclude -Itools/util/include -Iexamples/common \
    "dyy_test/op1/op1_${variant}.cu" -o "dyy_test/op1/bin/op1_${variant}"
done
```

Compile `op1_any_mnk.cu` with the same options to reproduce the documented
8192³ case. Its fixed random input is in `random_datasets_any_mnk`. For the
13-shape best executable, pass a writable dataset root and the historical
`--shape-id` and `--config-index` values in the result CSV. The random-data
harness supports `--generate-dataset --dataset-only`.
