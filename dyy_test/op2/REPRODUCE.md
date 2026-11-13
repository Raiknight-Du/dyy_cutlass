# OP2 reproduction

[Implementation and timing notes](README.md) · [Analysis](ANALYSIS_CN.md) · [Baseline versus best](results/op2_baseline_vs_best.csv) · [Equal-work correction](eq_corrected/ANALYSIS_CN.md) · [Roofline and figures](analysis/)

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
mkdir -p dyy_test/op2/bin
for variant in baseline best; do
  /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_100a \
    --expt-relaxed-constexpr -diag-suppress=20012 \
    -Iinclude -Itools/util/include -Iexamples/common \
    "dyy_test/op2/op2_${variant}.cu" -o "dyy_test/op2/bin/op2_${variant}"
done
```

For a standalone smoke run, invoke `dyy_test/op2/bin/op2_best --shape-id=13 --config-index=0 --iterations=10`.
The executable initializes deterministic random operands when `--dataset-root`
is omitted. The available archived input cases are in `random_datasets`.
Recover the rest of the original random datasets before replaying the full
historical 13-shape measurement with `--dataset-root`.
