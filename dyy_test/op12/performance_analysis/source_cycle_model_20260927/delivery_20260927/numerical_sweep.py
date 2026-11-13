#!/usr/bin/env python3
"""Deterministic source-calendar sweep from the frozen approved model only."""
from __future__ import annotations

import json
import hashlib
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "approved_model.json"
HARDWARE_PATH = HERE / "hardware_model.json"
OUT_JSON = HERE / "numerical_sweep.json"
OUT_MD = HERE / "numerical_sweep.md"


def ceil_div(a: float, b: float) -> int:
    return math.ceil(a / b) if a else 0


def dependencies(value):
    if isinstance(value, str):
        return value.split()
    return value or []


def rate_for(node, calendar, rates):
    resource = node["resource"]
    if resource == "HBM":
        return rates["hbm_bytes_per_sm_cycle_at_148"] * 148 / calendar.get("HBM_divisor", 148)
    if resource in ("HMMA", "TC16"):
        eta = calendar.get("eta_FP16", 1) if node.get("eta64", False) else 1
        return rates["fp16_flop_per_sm_cycle"] * eta
    if resource == "TF32":
        return rates["tf32_flop_per_sm_cycle"] * calendar.get("eta_TF32", 1)
    return rates.get(resource)


def schedule(calendar, rates):
    nodes = calendar["nodes"]
    by_id = {node["id"]: node for node in nodes}
    deps = {node["id"]: dependencies(node.get("dependency")) for node in nodes}
    # The stipulated resource calendar preserves the input source order for
    # successive jobs on each exclusive resource. Make that edge explicit so
    # an otherwise-ready later node cannot overtake an earlier source node.
    prior_on_resource = {}
    last_by_resource = {}
    for node in nodes:
        prior = last_by_resource.get(node["resource"])
        prior_on_resource[node["id"]] = prior
        if prior is not None and prior not in deps[node["id"]]:
            deps[node["id"]].append(prior)
        last_by_resource[node["resource"]] = node["id"]
    indegree = {node_id: len(parents) for node_id, parents in deps.items()}
    children = defaultdict(list)
    for node_id, parents in deps.items():
        for parent in parents:
            if parent not in by_id:
                raise ValueError(f"{calendar['key']}: missing predecessor {parent}")
            children[parent].append(node_id)
    # Stable Kahn order keeps source order for independent ready nodes.
    ready = deque(node["id"] for node in nodes if indegree[node["id"]] == 0)
    source_pos = {node["id"]: i for i, node in enumerate(nodes)}
    finishes, starts, services, by_resource_finish = {}, {}, {}, defaultdict(int)
    while ready:
        node_id = ready.popleft()
        node = by_id[node_id]
        resource = node["resource"]
        rate = rate_for(node, calendar, rates)
        demand = node["demand"]
        if demand and (rate is None or rate <= 0):
            raise ValueError(f"{calendar['key']}: no positive rate for {resource}")
        service = ceil_div(demand, rate) if demand else 0
        start = max([by_resource_finish[resource], *(finishes[p] for p in deps[node_id])])
        finish = start + service
        starts[node_id], services[node_id], finishes[node_id] = start, service, finish
        by_resource_finish[resource] = finish
        for child in sorted(children[node_id], key=source_pos.get):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(finishes) != len(nodes):
        unresolved = [node_id for node_id, degree in indegree.items() if degree]
        raise ValueError(f"{calendar['key']}: cycle or unresolved dependencies {unresolved[:5]}")
    resource_totals = defaultdict(float)
    unit_totals = defaultdict(float)
    resource_counts = Counter()
    resource_busy_cycles = defaultdict(int)
    for node in nodes:
        resource_totals[node["resource"]] += node["demand"]
        unit_totals[node["unit"]] += node["demand"]
        resource_counts[node["resource"]] += 1
        resource_busy_cycles[node["resource"]] += services[node["id"]]
    resource_spans = {}
    for resource in resource_counts:
        resource_spans[resource] = max(
            (finishes[n["id"]] for n in nodes if n["resource"] == resource), default=0
        )
    return {
        "calendar_key": calendar["key"],
        "scenario": {
            "HBM_divisor": calendar.get("HBM_divisor"),
            "eta_FP16": calendar.get("eta_FP16", 1),
            "eta_TF32": calendar.get("eta_TF32", 1),
        },
        "cycles": max(finishes.values(), default=0),
        "expected_cycles_from_approved_model": calendar.get("expected_cycles"),
        "delta_vs_expected_cycles": (
            max(finishes.values(), default=0) - calendar["expected_cycles"]
            if calendar.get("expected_cycles") is not None else None
        ),
        "nodes": len(nodes),
        "resource_node_counts": dict(resource_counts),
        "resource_demand": dict(resource_totals),
        "demand_by_unit": dict(unit_totals),
        "resource_work_floor_cycles": dict(resource_busy_cycles),
        "resource_control_totals": {
            "node_count": resource_counts.get("control", 0),
            "demand_total": resource_totals.get("control", 0),
            "demand_by_unit": {
                unit: sum(n["demand"] for n in nodes if n["resource"] == "control" and n["unit"] == unit)
                for unit in sorted({n["unit"] for n in nodes if n["resource"] == "control"})
            },
        },
        "resource_completion_cycles": resource_spans,
        "cta_span_cycles": max(finishes.values(), default=0),
        "node_timings": [
            {"id": n["id"], "resource": n["resource"], "demand": n["demand"],
             "unit": n["unit"], "start_cycle": starts[n["id"]],
             "service_cycles": services[n["id"]], "finish_cycle": finishes[n["id"]]}
            for n in nodes
        ],
    }


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    hw = json.loads(HARDWARE_PATH.read_text(encoding="utf-8"))
    device = hw["device_configuration"]
    peaks = hw["official_peaks"]
    sm = device["enabled_sms"]["value"]
    f = device["reference_sm_clock"]["value"]
    fp16_peak = peaks["dense_fp16_bf16"]["value"]
    tf32_peak = peaks["dense_tf32"]["value"]
    hbm_bw = peaks["hbm_bandwidth"]["value"]
    # Cross-check (not replace) the approved-model normalizations.
    source_rates = hw["scenario_a_rates"]
    peak_checks = {
        "fp16_source_rate": {
            "approved_hardware_rate": source_rates["tcgen05_fp16_bf16"]["value"],
            "derived_rate": fp16_peak / (sm * f),
        },
        "tf32_source_rate": {
            "approved_hardware_rate": source_rates["tcgen05_tf32"]["value"],
            "derived_rate": tf32_peak / (sm * f),
        },
        "hbm_equal_share_sm_rate": {
            "approved_hardware_rate": source_rates["hbm_equal_share_sm"]["value"],
            "derived_rate": hbm_bw / (sm * f),
        },
    }
    rates = {
        "fp16_flop_per_sm_cycle": fp16_peak / (sm * f),
        "tf32_flop_per_sm_cycle": tf32_peak / (sm * f),
        "hbm_bytes_per_sm_cycle_at_148": hbm_bw / (sm * f),
        "SMEM": 128,
        "TMEM": 512,
        "CVT": 64,
        "SIMT": 128,
        "ISSUE": 4,
    }
    for check in peak_checks.values():
        check["difference"] = check["derived_rate"] - check["approved_hardware_rate"]
        check["matches_within_1e-9_relative"] = math.isclose(
            check["derived_rate"], check["approved_hardware_rate"], rel_tol=1e-9
        )

    calendar_rows = []
    for c in model["calendars"]:
        calendar_rows.append(schedule(c, rates))
    by_key = {r["calendar_key"]: r for r in calendar_rows}
    calendars = {c["key"]: c for c in model["calendars"]}
    kernel_rows = []

    # K1 full-kernel sweep. Cbatch(b) is the source calendar for a synchronized
    # cohort of b CTAs sharing one SM resource schedule.
    cta_1 = by_key["K1_batch_1"]["cta_span_cycles"]
    cta_work = by_key["K1_batch_1"]["resource_demand"]
    for grid in (2048, 4096, 8192):
        q = ceil_div(grid, sm)
        full8, rem = divmod(q, 8)
        span = full8 * by_key["K1_batch_8"]["cycles"]
        if rem:
            span += by_key[f"K1_batch_{rem}"]["cycles"]
        gpu_hbm_floor = ceil_div(grid * 24840 * f, hbm_bw)
        kernel_cycles = max(span, gpu_hbm_floor)
        kernel_rows.append({
            "kernel": "K1", "configuration": f"grid{grid}", "calendar_key": "K1_batch_1..8",
            "grid": grid, "residency_cta_per_sm": 8, "sm_count": sm,
            "waves": ceil_div(q, 8), "wave_interpretation": "per-SM cohorts of up to 8 CTAs; K1 formula aggregates floor(q/8) full cohorts plus q mod 8 remainder",
            "cta_span_cycles": cta_1, "cta_cohort8_span_cycles": by_key["K1_batch_8"]["cycles"],
            "sm_cohort_cycles": span, "gpu_hbm_floor_cycles": gpu_hbm_floor,
            "kernel_cycles": kernel_cycles, "time_us": kernel_cycles / f * 1e6,
            "resource_demand_per_cta": dict(cta_work),
            "resource_demand_kernel": {k: v * grid for k, v in cta_work.items()},
            "global_hbm_traffic_bytes": cta_work.get("HBM", 0) * grid,
            "flops_per_cta": {k: v for k, v in cta_work.items() if k in ("HMMA", "TC16", "TF32")},
            "flops_kernel": {k: v * grid for k, v in cta_work.items() if k in ("HMMA", "TC16", "TF32")},
            "model": "max(floor(q/8)*Cbatch8 + Cbatch(q mod 8), ceil(grid*24840*f/HBM_BW)); q=ceil(grid/SM)",
        })

    # One balanced CTA wave, one CTA per participating SM for K2/K4 and K3 P8.
    base_specs = [
        ("K2", f"K2_{l}", 128, 1, l) for l in (16, 32, 64)
    ] + [("K3P8", "K3_8", 32, 1, 8)] + [
        ("K4", f"K4_{l}", 128, 1, l) for l in (16, 32, 64)
    ]
    for kernel, key, grid, residency, length in base_specs:
        row = by_key[key]
        work = row["resource_demand"]
        kernel_rows.append({
            "kernel": kernel, "configuration": f"L{length}", "calendar_key": key,
            "grid": grid, "residency_cta_per_sm": residency, "sm_count": sm,
            "waves": ceil_div(grid, sm * residency),
            "wave_interpretation": "one balanced wave; calendar is one CTA span and global resources are equal-share per-SM as configured",
            "cta_span_cycles": row["cta_span_cycles"], "kernel_cycles": row["cta_span_cycles"],
            "time_us": row["cta_span_cycles"] / f * 1e6,
            "resource_demand_per_cta": dict(work),
            "resource_demand_kernel": {k: v * grid for k, v in work.items()},
            "global_hbm_traffic_bytes": work.get("HBM", 0) * grid,
            "flops_per_cta": {k: v for k, v in work.items() if k in ("HMMA", "TC16", "TF32")},
            "flops_kernel": {k: v * grid for k, v in work.items() if k in ("HMMA", "TC16", "TF32")},
        })

    # All approved sensitivity calendars are retained as distinct rows.
    for key, kernel, grid, residency, config in [
        ("K3_8_div32", "K3P8", 32, 1, "L8_HBMdiv32"),
        ("K3_8_etaHalf", "K3P8", 32, 1, "L8_etaTF32_0.5"),
        ("K4_16_div128", "K4", 128, 1, "L16_HBMdiv128"),
        ("K4_16_etaHalf", "K4", 128, 1, "L16_eta64_0.5"),
    ]:
        row = by_key[key]
        work = row["resource_demand"]
        kernel_rows.append({
            "kernel": kernel, "configuration": config, "calendar_key": key,
            "grid": grid, "residency_cta_per_sm": residency, "sm_count": sm,
            "waves": ceil_div(grid, sm * residency),
            "wave_interpretation": "one balanced wave; sensitivity kept separate from base calendar",
            "cta_span_cycles": row["cta_span_cycles"], "kernel_cycles": row["cta_span_cycles"],
            "time_us": row["cta_span_cycles"] / f * 1e6,
            "resource_demand_per_cta": dict(work),
            "resource_demand_kernel": {k: v * grid for k, v in work.items()},
            "global_hbm_traffic_bytes": work.get("HBM", 0) * grid,
            "flops_per_cta": {k: v for k, v in work.items() if k in ("HMMA", "TC16", "TF32")},
            "flops_kernel": {k: v * grid for k, v in work.items() if k in ("HMMA", "TC16", "TF32")},
        })

    kernel_rows.sort(key=lambda r: (r["kernel"], r["configuration"]))
    trends = []
    for kernel in ("K2", "K4"):
        rows = [r for r in kernel_rows if r["kernel"] == kernel and r["configuration"] in ("L16", "L32", "L64")]
        trends.append({
            "kernel": kernel,
            "lengths": [int(r["configuration"][1:]) for r in rows],
            "cycles": [r["kernel_cycles"] for r in rows],
            "time_us": [r["time_us"] for r in rows],
            "hbm_bytes_per_cta": [r["resource_demand_per_cta"].get("HBM", 0) for r in rows],
            "fp16_flops_per_cta": [r["resource_demand_per_cta"].get("TC16", 0) + r["resource_demand_per_cta"].get("HMMA", 0) for r in rows],
        })
    trends.append({
        "kernel": "K1",
        "grids": [r["grid"] for r in kernel_rows if r["kernel"] == "K1"],
        "cycles": [r["kernel_cycles"] for r in kernel_rows if r["kernel"] == "K1"],
        "time_us": [r["time_us"] for r in kernel_rows if r["kernel"] == "K1"],
    })
    violations = [
        {"calendar_key": r["calendar_key"], "expected_cycles": r["expected_cycles_from_approved_model"],
         "computed_cycles": r["cycles"], "delta": r["delta_vs_expected_cycles"]}
        for r in calendar_rows if r["delta_vs_expected_cycles"] not in (None, 0)
    ]
    output = {
        "title": "Approved source-calendar numerical sweep",
        "status": "FINAL_SOURCE_ONLY_INDEPENDENT_REPLAY",
        "provenance": {
            "approved_model": str(MODEL_PATH), "hardware_model": str(HARDWARE_PATH),
            "approved_model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "hardware_model_sha256": hashlib.sha256(HARDWARE_PATH.read_bytes()).hexdigest(),
            "measurement_files_read": False,
            "unknown_tails": "zero numeric extra service cost; source dependencies preserved",
            "scheduler": "stable topological list schedule; node service ceil(demand/rate); start=max(predecessor finishes, previous job finish on same resource)",
            "resource_rates": {
                "SM_count": sm, "reference_clock_Hz": f,
                "FP16_FLOP_per_SM_cycle": rates["fp16_flop_per_sm_cycle"],
                "TF32_FLOP_per_SM_cycle": rates["tf32_flop_per_sm_cycle"],
                "HBM_byte_per_SM_cycle_at_divisor148": rates["hbm_bytes_per_sm_cycle_at_148"],
                "SMEM_byte_per_cycle": rates["SMEM"], "TMEM_byte_per_cycle": rates["TMEM"],
                "CVT_elements_per_cycle": rates["CVT"], "SIMT_results_per_cycle": rates["SIMT"],
                "ISSUE_warp_instructions_per_cycle": rates["ISSUE"],
            },
            "source_peak_normalization_checks": peak_checks,
            "timeline_rate_rule": "HBM rate = 7.7e12 / (f * HBM_divisor) bytes per SM reference cycle; eta_FP16 applied only to nodes marked eta64; eta_TF32 to TF32 nodes.",
            "source_review_violations_preserved": model.get("review", []),
        },
        "calendar_results": calendar_rows,
        "kernel_results": kernel_rows,
        "trends": trends,
        "expected_cycle_differences_preserved": violations,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Final source-calendar numerical sweep", "",
        "**Status: FINAL_SOURCE_ONLY_INDEPENDENT_REPLAY.**", "",
        "Inputs: normalized `approved_model.json` and `hardware_model.json` only. No NCU/event/target timing files were inspected.",
        "Unknown tails have explicit zero numeric service cost, while every source dependency remains in the schedule.",
        "Each node receives `ceil(demand/rate)` cycles; it starts at the maximum predecessor finish and previous finish on that resource.",
        "The 19 normalized source calendars are each reported below and checked against their regenerated expected-cycle values.", "",
        "Rates use FP16/TF32 official dense peaks divided by `SM*f`; HBM uses `7.7e12/(f*HBM_divisor)` bytes per SM cycle. The computed FP16, TF32 and divisor-148 HBM rates are cross-checked against `hardware_model.json`.", "",
        "## Full kernel sweep", "",
        "| Kernel | Config | Calendar | Grid | Residency | Waves | CTA span (cy) | Kernel (cy) | Time (µs) | HBM bytes/kernel | FP16 FLOPs/kernel | TF32 FLOPs/kernel |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in kernel_rows:
        flop = r["flops_kernel"]
        lines.append(
            f"| {r['kernel']} | {r['configuration']} | {r['calendar_key']} | {r['grid']} | {r['residency_cta_per_sm']} | {r['waves']} | {r['cta_span_cycles']} | {r['kernel_cycles']} | {r['time_us']:.6f} | {r['global_hbm_traffic_bytes']:.0f} | {flop.get('TC16', 0)+flop.get('HMMA', 0):.0f} | {flop.get('TF32', 0):.0f} |"
        )
    lines += ["", "Wave interpretation: K1 uses `q=ceil(grid/148)` and batches cohorts of up to eight CTAs per SM; its total is `max(floor(q/8)*Cbatch8 + Cbatch(q mod 8), ceil(grid*24840*f/7.7e12))`. K2, K4, and K3 P8 use their instructed one balanced wave, with kernel cycles equal to the one-CTA span. Sensitivity calendars are displayed separately.", "", "## Calendar audit", "", "Resource-work floor is the maximum sum of service cycles for any one resource in the calendar; DAG span additionally includes dependency-caused idle gaps.", "", "| Calendar | Resource-work floor (cy) | DAG span (cy) | Expected (cy) | Delta | HBM bytes | SMEM bytes | TMEM bytes | FP16/TC16 FLOPs | TF32 FLOPs | SIMT results | CVT elements | ISSUE warp-inst | Control nodes / demand |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in calendar_rows:
        d = r["resource_demand"]
        floor = max(r["resource_work_floor_cycles"].values(), default=0)
        ctrl = r["resource_control_totals"]
        lines.append(
            f"| {r['calendar_key']} | {floor} | {r['cycles']} | {r['expected_cycles_from_approved_model'] if r['expected_cycles_from_approved_model'] is not None else '—'} | {r['delta_vs_expected_cycles'] if r['delta_vs_expected_cycles'] is not None else '—'} | {d.get('HBM',0):.0f} | {d.get('SMEM',0):.0f} | {d.get('TMEM',0):.0f} | {d.get('HMMA',0)+d.get('TC16',0):.0f} | {d.get('TF32',0):.0f} | {d.get('SIMT',0):.0f} | {d.get('CVT',0):.0f} | {d.get('ISSUE',0):.0f} | {ctrl['node_count']} / {ctrl['demand_total']:.0f} |"
        )
    lines += ["", "## Trend readout", ""]
    for t in trends:
        if t["kernel"] in ("K2", "K4"):
            lines.append(f"- {t['kernel']} L16/32/64 cycles: " + " / ".join(map(str, t["cycles"])) + "; time (µs): " + " / ".join(f"{v:.6f}" for v in t["time_us"]) + ".")
        else:
            lines.append(f"- K1 grids 2048/4096/8192 cycles: " + " / ".join(map(str, t["cycles"])) + "; time (µs): " + " / ".join(f"{v:.6f}" for v in t["time_us"]) + ".")
    lines += ["", "## Unreconciled differences", ""]
    if violations:
        lines.append("Computed calendar spans differ from the approved JSON's stored expected-cycle values. These differences are retained verbatim for review:")
        for v in violations:
            lines.append(f"- `{v['calendar_key']}`: expected {v['expected_cycles']}, recomputed {v['computed_cycles']}, delta {v['delta']} cycles.")
    else:
        lines.append("All 19 independently recomputed calendar spans match the regenerated approved expected-cycle values; there are no numeric disagreements.")
    lines += ["", "## Full per-kernel resource totals", "", "Demands below include total work across each reported grid; K1 uses the per-CTA calendar demands multiplied by grid. `control` demand remains zero and all control-node counts/demand-by-unit are retained in the JSON calendar records.", "", "| Kernel | Config | HBM bytes | SMEM bytes | TMEM bytes | HMMA FLOPs | TC16 FLOPs | TF32 FLOPs | SIMT results | CVT elements | ISSUE warp-inst | Control demand |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in kernel_rows:
        d = r["resource_demand_kernel"]
        lines.append(f"| {r['kernel']} | {r['configuration']} | {d.get('HBM',0):.0f} | {d.get('SMEM',0):.0f} | {d.get('TMEM',0):.0f} | {d.get('HMMA',0):.0f} | {d.get('TC16',0):.0f} | {d.get('TF32',0):.0f} | {d.get('SIMT',0):.0f} | {d.get('CVT',0):.0f} | {d.get('ISSUE',0):.0f} | {d.get('control',0):.0f} |")
    lines += ["", "Unknown latency/issue/SFU/barrier tails remain unpriced at zero numeric cost, so these are explicit ideal-tail source schedules, not measured runtimes or lower-bound claims. Source-level review issues from `approved_model.json` remain recorded in the JSON output.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
