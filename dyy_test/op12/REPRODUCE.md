# OP12 reproduction

- [Algorithm and upstream audit](README.md) and [source/environment pins](SOURCES.json)
- [GDN analysis report](analysis/昇腾竞争力benchmark构建性能分析报告-GDN.docx), [Roofline workbook](analysis/OP12_ROOFLINE.xlsx) and [cycle model](analysis/OP12_FOUR_KERNEL_SOURCE_CYCLE_MODEL_20260927_INTEGER_RATES_REV2.xlsx)
- [Frozen CP4 Event baseline](performance_analysis/cp4_event_baseline_freeze_20260929/FROZEN_EVENT_BASELINE_CN.md) and [GPC frequency timelines](performance_analysis/cp4_classified_results_20261023/README.md)
- [FP8 experiments](implementation/README.md) and [native CUDA GDN](native_cuda/README.md)

Restore the pinned FlashInfer checkout with its submodules under
`dyy_test/op12/upstreams/flashinfer`. Set up the CUDA 13 and CuTe DSL
environment described in [SMOKE_RESULTS.md](validation/SMOKE_RESULTS.md).

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git dyy_test/op12/upstreams/flashinfer
git -C dyy_test/op12/upstreams/flashinfer checkout e77a4a0d276367895c3b50a642fd8f326c03fb72
git -C dyy_test/op12/upstreams/flashinfer submodule update --init --recursive
PYTHONPATH="$PWD/dyy_test/op12/upstreams/flashinfer${PYTHONPATH:+:$PYTHONPATH}" \
  python dyy_test/op12/validation/run_smoke.py
```

The CP4 Event runner regenerates its inputs from the shape-specific seed.
For one short diagnostic run with a fresh output directory:

```bash
python dyy_test/op12/performance_analysis/cp4_event_baseline_freeze_20260929/runner/measure_cp4_event.py \
  --repo "$PWD/dyy_test/op12/upstreams/flashinfer" \
  --output-root "$(mktemp -d)" --s 8192 --trial 1 --pilot
```

The performance archive includes the published measurement tables, analysis
scripts, and selected K1–K4 cases. To rerun a historical ablation that
consumes an exact frozen input, large profiler capture or generated
intermediate, restore it at the path recorded in the experiment manifest.
Original material is under
`/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12` on B200 and in
`OP12_实验归档与分析报告_20261110` in the local benchmark archive.
