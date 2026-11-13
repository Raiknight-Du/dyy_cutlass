#!/usr/bin/env python3
"""Aggregate three native timing trials and compare with pinned FlashInfer.

The input JSONL files are emitted by ``benchmarks/benchmark_native_gdn.py``.
The optional FlashInfer CSV is the same-method summary produced by the op12
fixed-commit performance audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if obj.get("kind") == "case_result" and obj.get("status") == "PASS":
                rows.append(obj)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)

    result: list[dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        exemplar = items[0]
        medians = [float(x["latency_ms_median"]) for x in items]
        median_ms = statistics.median(medians)
        mean_ms = statistics.fmean(medians)
        cross_cv = (
            statistics.pstdev(medians) / mean_ms * 100.0 if mean_ms else 0.0
        )
        heads = exemplar["heads"]
        upstream_flops = (
            4 * exemplar["total_tokens"] * heads[2] * 128 * 128
        )
        seven_flops = (
            exemplar["logical_chunks"]
            * exemplar["output_heads"]
            * 10_485_760
        )
        backends = sorted({str(x["effective_backend"]) for x in items})
        result.append(
            {
                "name": name,
                "category": exemplar["category"],
                "dtype": exemplar["dtype"],
                "seq_lens": ",".join(map(str, exemplar["seq_lens"])),
                "heads": "/".join(map(str, heads)),
                "use_cp": str(exemplar["use_cp"]),
                "predicted_dispatch": exemplar["predicted_dispatch"],
                "effective_backend": "+".join(backends),
                "batch": exemplar["batch"],
                "total_tokens": exemplar["total_tokens"],
                "output_heads": exemplar["output_heads"],
                "non_cp_grid": exemplar["non_cp_grid"],
                "waves_per_sm_non_cp": exemplar["waves_per_sm_non_cp"],
                "trial_count": len(items),
                "median_of_medians_ms": median_ms,
                "min_trial_median_ms": min(medians),
                "max_trial_median_ms": max(medians),
                "cross_trial_cv_pct": cross_cv,
                "tokens_per_second": exemplar["total_tokens"] / (median_ms / 1e3),
                "upstream_work_tflops": upstream_flops / (median_ms / 1e3) / 1e12,
                "seven_gemm_tflops": seven_flops / (median_ms / 1e3) / 1e12,
                "max_within_trial_cv_pct": max(
                    float(x["latency_cv_pct"]) for x in items
                ),
                "all_finite": all(
                    x["finite_output"] and x["finite_state"] for x in items
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no rows to write")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(
    native: list[dict[str, Any]], flashinfer_csv: Path
) -> list[dict[str, Any]]:
    with flashinfer_csv.open(encoding="utf-8-sig") as stream:
        flashinfer = {row["name"]: row for row in csv.DictReader(stream)}
    rows: list[dict[str, Any]] = []
    for nrow in native:
        frow = flashinfer.get(nrow["name"])
        if frow is None:
            continue
        nms = float(nrow["median_of_medians_ms"])
        fms = float(frow["median_of_medians_ms"])
        rows.append(
            {
                "name": nrow["name"],
                "category": nrow["category"],
                "dtype": nrow["dtype"],
                "use_cp": nrow["use_cp"],
                "native_backend": nrow["effective_backend"],
                "native_ms": nms,
                "flashinfer_ms": fms,
                "native_over_flashinfer_latency_ratio": nms / fms,
                "flashinfer_over_native_speedup": nms / fms,
                "native_tokens_per_second": nrow["tokens_per_second"],
                "flashinfer_tokens_per_second": frow["tokens_per_second"],
                "native_all_finite": nrow["all_finite"],
                "native_trial_count": nrow["trial_count"],
                "flashinfer_trial_count": frow["trial_count"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flashinfer-summary", type=Path)
    args = parser.parse_args()

    native = aggregate(load_jsonl(args.input))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "benchmark_native_summary.csv", native)
    (out / "benchmark_native_summary.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.flashinfer_summary:
        compared = comparison(native, args.flashinfer_summary.resolve())
        write_csv(out / "benchmark_native_vs_flashinfer.csv", compared)
        (out / "benchmark_native_vs_flashinfer.json").write_text(
            json.dumps(compared, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    incomplete = [row["name"] for row in native if row["trial_count"] != 3]
    if incomplete:
        raise RuntimeError(f"expected 3 trials, incomplete cases: {incomplete}")
    if not all(bool(row["all_finite"]) for row in native):
        raise RuntimeError("one or more native cases produced NaN/Inf")
    print(f"cases={len(native)} trials={len(args.input)} output={out}")


if __name__ == "__main__":
    main()
