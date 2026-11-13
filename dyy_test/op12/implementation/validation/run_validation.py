#!/usr/bin/env python3
"""Run op12 correctness cases and emit machine-readable plus Markdown results."""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from dataclasses import dataclass, asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from op12_gdn import gdn_prefill_fp8


@dataclass
class Result:
    name: str
    passed: bool
    o_relative_l2: float
    o_cosine: float
    state_relative_l2: float
    max_saturation_rate: float
    details: str


def fp8_randn(shape, amplitude=0.05):
    return (torch.randn(shape, device="cuda") * amplitude).to(torch.float8_e4m3fn)


def metric(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    rel = float((af - bf).norm() / bf.norm().clamp_min(1.0e-12))
    cos = 1.0 if af.numel() == 0 else float(torch.nn.functional.cosine_similarity(af, bf, dim=0))
    return rel, cos


def max_saturation(stats):
    values = [x["max_saturation_rate"] for x in stats.get("quantization", {}).values()]
    return max(values, default=0.0)


def evaluate(name, args, kwargs):
    (o, state), stats = gdn_prefill_fp8(*args, **kwargs, output_final_state=True, return_stats=True)
    ref_o, ref_state = gdn_prefill_fp8(*args, **kwargs, output_final_state=True, backend="reference")
    o_rel, o_cos = metric(o, ref_o)
    s_rel, _ = metric(state, ref_state)
    finite = bool(torch.isfinite(o).all().item() and torch.isfinite(state).all().item())
    passed = finite and o_rel <= 0.20 and o_cos >= 0.98 and s_rel <= 0.25
    return Result(name, passed, o_rel, o_cos, s_rel, max_saturation(stats), json.dumps(stats, ensure_ascii=False))


def main():
    torch.manual_seed(12)
    results = []
    for seq_len in (1, 63, 64, 65, 127, 257):
        q, k, v = (fp8_randn((1, 1, seq_len, 128)) for _ in range(3))
        gate = torch.empty((1, 1, seq_len), dtype=torch.float32, device="cuda").uniform_(0.96, 1.0)
        beta = torch.empty_like(gate).uniform_(0.05, 0.3)
        results.append(evaluate(f"BNSD_S{seq_len}", (q, k, v, gate, beta), {"use_cp": False}))

    lengths = [1, 63, 64, 65, 127, 257]
    total = sum(lengths)
    q, k, v = (fp8_randn((total, 1, 128)) for _ in range(3))
    gate = torch.empty((total, 1), dtype=torch.float32, device="cuda").uniform_(0.97, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.25)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    results.append(evaluate("TND_varlen", (q, k, v, gate, beta), {"cu_seqlens": cu, "use_cp": False}))

    for name, (hq, hk, hv) in (("GQA_8_2", (8, 2, 2)), ("GVA_2_8", (2, 2, 8))):
        seq_len, ho = 64, max(hq, hv)
        q, k, v = fp8_randn((seq_len, hq, 128)), fp8_randn((seq_len, hk, 128)), fp8_randn((seq_len, hv, 128))
        gate = torch.empty((seq_len, ho), dtype=torch.float32, device="cuda").uniform_(0.98, 1.0)
        beta = torch.empty_like(gate).uniform_(0.05, 0.2)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        results.append(evaluate(name, (q, k, v, gate, beta), {"cu_seqlens": cu, "use_cp": False}))

    seq_len = 65
    q, k, v = (fp8_randn((seq_len, 1, 128)) for _ in range(3))
    gate = torch.empty((seq_len, 1), dtype=torch.float32, device="cuda").uniform_(0.98, 1.0)
    beta = torch.empty_like(gate).uniform_(0.05, 0.2)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    initial = torch.randn((1, 1, 128, 128), dtype=torch.float16, device="cuda") * 0.01
    results.append(evaluate("TND_initial_state", (q, k, v, gate, beta), {"cu_seqlens": cu, "initial_state": initial, "use_cp": False}))

    cp_len = 4096
    q, k, v = (fp8_randn((cp_len, 1, 128), 0.01) for _ in range(3))
    gate = torch.full((cp_len, 1), 0.9999, dtype=torch.float32, device="cuda")
    beta = torch.full_like(gate, 0.05)
    cu = torch.tensor([0, cp_len], dtype=torch.int32, device="cuda")
    (cp_o, cp_s), cp_stats = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp="auto", return_stats=True)
    base_o, base_s = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False)
    o_rel, o_cos = metric(cp_o, base_o)
    s_rel, _ = metric(cp_s, base_s)
    results.append(Result("CP_semantic_4096", cp_stats["route"] == "cp_semantic_4096" and o_rel == 0.0 and s_rel == 0.0, o_rel, o_cos, s_rel, max_saturation(cp_stats), json.dumps(cp_stats, ensure_ascii=False)))

    negative = []
    try:
        gdn_prefill_fp8(q.float(), k, v, gate, beta, cu_seqlens=cu)
    except ValueError as exc:
        negative.append(f"FP8 Q contract rejection: {exc}")
    try:
        bq = fp8_randn((1, 1, 16, 128))
        bg = torch.ones((1, 1, 16), dtype=torch.float32, device="cuda")
        gdn_prefill_fp8(bq, bq, bq, bg, bg, cu_seqlens=torch.tensor([0, 16], dtype=torch.int32, device="cuda"))
    except ValueError as exc:
        negative.append(f"BNSD cu_seqlens rejection: {exc}")

    payload = {
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
        "thresholds": {"o_relative_l2_max": 0.20, "o_cosine_min": 0.98, "state_relative_l2_max": 0.25},
        "results": [asdict(r) for r in results],
        "negative_tests": negative,
        "all_passed": all(r.passed for r in results) and len(negative) == 2,
    }
    out_dir = ROOT / "validation"
    (out_dir / "implementation_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [
        "# op12 B200 实现验证结果",
        "",
        f"- 环境：{payload['environment']['gpu']} / CC {payload['environment']['compute_capability']} / PyTorch {payload['environment']['torch']} / CUDA {payload['environment']['cuda_runtime']}",
        f"- 总结：{'全部通过' if payload['all_passed'] else '存在失败'}（{sum(r.passed for r in results)}/{len(results)} accuracy cases）",
        "- 注意：本结果针对 unfused seven-GEMM backend；不代表 2-SM cluster 或 CP parallel-prefix 已完成。",
        "",
        "| case | pass | O rel-L2 | O cosine | state rel-L2 | max saturation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rows.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.o_relative_l2:.6f} | {r.o_cosine:.6f} | {r.state_relative_l2:.6f} | {r.max_saturation_rate:.6f} |")
    rows += ["", "## 负向测试", ""] + [f"- {x}" for x in negative]
    (out_dir / "IMPLEMENTATION_RESULTS.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["all_passed"] else 1)


if __name__ == "__main__":
    main()
