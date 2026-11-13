#!/usr/bin/env python3
"""Validate direct-cast/unit-scale FP8 GDN against FlashInfer's own reference.

The algorithm, block size, input generator, head expansion, gate/beta semantics,
and FP32 gold are taken from the archived FlashInfer source tree.  The tested
backend changes only the seven logical GEMM operand dtypes to E4M3.  All MMA
operand scales are literal one; no amax-derived scale or dequant multiply is
used.  The algorithmic query scale (1/sqrt(128)) remains unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OP12_ROOT = ROOT.parent
UPSTREAM = OP12_ROOT / "upstreams" / "flashinfer"
sys.path.insert(0, str(ROOT))

import torch

from op12_gdn import gdn_prefill_fp8


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref_mod = _load_module(
    "flashinfer_reference_delta_rule",
    UPSTREAM / "tests" / "gdn" / "reference_delta_rule.py",
)
fixture_mod = _load_module(
    "flashinfer_gdn_conftest",
    UPSTREAM / "tests" / "gdn" / "conftest.py",
)


@dataclass
class Metrics:
    o_relative_l2: float
    o_cosine: float
    o_max_abs: float
    state_relative_l2: float
    state_cosine: float
    state_max_abs: float


@dataclass
class Result:
    name: str
    layout: str
    seq_lens: list[int]
    heads: list[int]
    qv_multiplier: float
    initial_state: bool
    finite: bool
    op12_pass: bool
    upstream_strict_allclose: bool
    no_scale: Metrics
    tile_scaled: Metrics
    max_saturation_rate: float
    max_zeroed_nonzero_rate: float
    quantization: dict[str, Any]


def metric(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    rel = float((af - bf).norm() / bf.norm().clamp_min(1.0e-12))
    cos = float(torch.nn.functional.cosine_similarity(af, bf, dim=0))
    max_abs = float((af - bf).abs().max())
    return rel, cos, max_abs


def metrics(
    out: torch.Tensor,
    state: torch.Tensor,
    ref_out: torch.Tensor,
    ref_state: torch.Tensor,
) -> Metrics:
    o_rel, o_cos, o_max = metric(out, ref_out)
    s_rel, s_cos, s_max = metric(state, ref_state)
    return Metrics(o_rel, o_cos, o_max, s_rel, s_cos, s_max)


def make_inputs(
    seq_lens: list[int],
    heads: tuple[int, int, int],
    *,
    qv_multiplier: float,
    layout: str,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    hq, hk, hv = heads
    q, k, v = fixture_mod.gen_qkv(seq_lens, hq, hk, hv, 128, torch.float32)
    q = (q * qv_multiplier).to(device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.nn.functional.normalize(k.to(device="cuda"), p=2.0, dim=-1).to(
        torch.float8_e4m3fn
    )
    v = (v * qv_multiplier).to(device="cuda", dtype=torch.float8_e4m3fn)
    ho = max(hq, hv)
    total = sum(seq_lens)
    gate = torch.rand((total, ho), dtype=torch.float32, device="cuda")
    beta = torch.rand((total, ho), dtype=torch.float32, device="cuda")
    cu = torch.tensor(
        ref_mod.exclusive_cumsum(seq_lens), dtype=torch.int32, device="cuda"
    )
    if layout == "TND":
        return q, k, v, gate, beta, cu
    if len(set(seq_lens)) != 1:
        raise ValueError("BNSD requires equal sequence lengths")
    batch, seq = len(seq_lens), seq_lens[0]
    q = q.reshape(batch, seq, hq, 128).permute(0, 2, 1, 3).contiguous()
    k = k.reshape(batch, seq, hk, 128).permute(0, 2, 1, 3).contiguous()
    v = v.reshape(batch, seq, hv, 128).permute(0, 2, 1, 3).contiguous()
    gate = gate.reshape(batch, seq, ho).permute(0, 2, 1).contiguous()
    beta = beta.reshape(batch, seq, ho).permute(0, 2, 1).contiguous()
    return q, k, v, gate, beta, None


def to_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
):
    if q.ndim == 3:
        return q, k, v, gate, beta
    batch, hq, seq, dim = q.shape
    hk, hv, ho = k.shape[1], v.shape[1], gate.shape[1]
    assert seq_lens == [seq] * batch
    return (
        q.permute(0, 2, 1, 3).reshape(-1, hq, dim),
        k.permute(0, 2, 1, 3).reshape(-1, hk, dim),
        v.permute(0, 2, 1, 3).reshape(-1, hv, dim),
        gate.permute(0, 2, 1).reshape(-1, ho),
        beta.permute(0, 2, 1).reshape(-1, ho),
    )


def from_tnd_output(ref_out: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 3:
        return ref_out
    batch, _, seq, dim = q.shape
    ho = ref_out.shape[1]
    return ref_out.reshape(batch, seq, ho, dim).permute(0, 2, 1, 3).contiguous()


def max_stat(stats: dict[str, Any], key: str) -> float:
    return max((float(v.get(key, 0.0)) for v in stats.values()), default=0.0)


def evaluate(
    name: str,
    seq_lens: list[int],
    heads: tuple[int, int, int] = (1, 1, 1),
    *,
    layout: str = "TND",
    qv_multiplier: float = 1.0,
    with_initial_state: bool = False,
    seed: int = 1200,
) -> Result:
    q, k, v, gate, beta, cu = make_inputs(
        seq_lens,
        heads,
        qv_multiplier=qv_multiplier,
        layout=layout,
        seed=seed,
    )
    ho = max(heads[0], heads[2])
    initial = None
    if with_initial_state:
        initial = (
            torch.randn(
                (len(seq_lens), ho, 128, 128),
                dtype=torch.float32,
                device="cuda",
            )
            * 0.01
        ).to(torch.float16)

    kwargs: dict[str, Any] = {
        "output_final_state": True,
        "use_cp": False,
        "initial_state": initial,
    }
    if cu is not None:
        kwargs["cu_seqlens"] = cu

    (out, state), stats = gdn_prefill_fp8(
        q,
        k,
        v,
        gate,
        beta,
        backend="unit_scale_fp8",
        return_stats=True,
        **kwargs,
    )
    scaled_out, scaled_state = gdn_prefill_fp8(
        q,
        k,
        v,
        gate,
        beta,
        backend="scaled_mm",
        **kwargs,
    )

    qt, kt, vt, gt, bt = to_tnd(q, k, v, gate, beta, seq_lens)
    ref_initial = None if initial is None else initial.transpose(-1, -2).float()
    ref_out_tnd, ref_state_kv = ref_mod.blockwise_delta_rule(
        qt.float(),
        kt.float(),
        vt.float(),
        seq_lens,
        alpha=gt,
        beta=bt,
        block_size=64,
        scale_factor=1.0 / math.sqrt(128),
        state_dtype=torch.float32,
        initial_state=ref_initial,
    )
    ref_out = from_tnd_output(ref_out_tnd, q)
    ref_state = ref_state_kv.transpose(-1, -2).contiguous()
    no_scale_metrics = metrics(out, state, ref_out, ref_state)
    scaled_metrics = metrics(scaled_out, scaled_state, ref_out, ref_state)
    finite = bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    op12_pass = (
        finite
        and no_scale_metrics.o_relative_l2 <= 0.20
        and no_scale_metrics.o_cosine >= 0.98
        and no_scale_metrics.state_relative_l2 <= 0.25
    )
    strict = bool(
        torch.allclose(out.float(), ref_out.float(), atol=2.0e-3, rtol=1.0e-3)
        and torch.allclose(
            state.float(), ref_state.float(), atol=1.0e-3, rtol=1.0e-4
        )
    )
    qstats = stats.get("quantization", {})
    return Result(
        name=name,
        layout=layout,
        seq_lens=seq_lens,
        heads=list(heads),
        qv_multiplier=qv_multiplier,
        initial_state=with_initial_state,
        finite=finite,
        op12_pass=op12_pass,
        upstream_strict_allclose=strict,
        no_scale=no_scale_metrics,
        tile_scaled=scaled_metrics,
        max_saturation_rate=max_stat(qstats, "max_saturation_rate"),
        max_zeroed_nonzero_rate=max_stat(qstats, "max_zeroed_nonzero_rate"),
        quantization=qstats,
    )


def main() -> None:
    cases = []
    for seq in (1, 63, 64, 65, 127, 257):
        cases.append(
            evaluate(
                f"BNSD_S{seq}",
                [seq],
                layout="BNSD",
                seed=1200 + seq,
            )
        )
    varlen = [1, 63, 64, 65, 127, 257]
    cases.append(evaluate("TND_varlen", varlen, seed=1300))
    cases.append(evaluate("GQA_8_2", [64], (8, 2, 2), seed=1301))
    cases.append(evaluate("GVA_2_8", [64], (2, 2, 8), seed=1302))
    cases.append(
        evaluate("TND_initial_state", [65], with_initial_state=True, seed=1303)
    )
    cases.append(
        evaluate("QKV_low_0p25", [64], qv_multiplier=0.25, seed=1304)
    )
    cases.append(
        evaluate("QKV_high_4p0", [64], qv_multiplier=4.0, seed=1305)
    )
    cases.append(evaluate("TND_long_4096", [4096], seed=1306))

    upstream_head = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "scope": {
            "upstream": "https://github.com/flashinfer-ai/flashinfer",
            "upstream_commit": upstream_head,
            "algorithm_reference": "tests/gdn/reference_delta_rule.py::blockwise_delta_rule",
            "input_generator": "tests/gdn/conftest.py::gen_qkv",
            "changed_dimension": "seven logical GEMM operands use E4M3",
            "quantization_scale": "none; direct saturating E4M3 cast",
            "mma_operand_scales": "literal one required by torch._scaled_mm ABI",
            "algorithmic_qk_scale": "1/sqrt(128), retained from upstream",
            "accumulator": "FP32",
            "output_state": "FP16",
            "backend": "unfused torch._scaled_mm correctness vehicle",
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(0))
            ),
        },
        "thresholds": {
            "op12": {
                "o_relative_l2_max": 0.20,
                "o_cosine_min": 0.98,
                "state_relative_l2_max": 0.25,
            },
            "upstream_fp16_strict": {
                "o_atol": 0.002,
                "o_rtol": 0.001,
                "state_atol": 0.001,
                "state_rtol": 0.0001,
            },
        },
        "results": [asdict(case) for case in cases],
        "summary": {
            "op12_passed": sum(case.op12_pass for case in cases),
            "total": len(cases),
            "all_op12_passed": all(case.op12_pass for case in cases),
            "upstream_strict_passed": sum(
                case.upstream_strict_allclose for case in cases
            ),
        },
    }
    out_dir = ROOT / "validation"
    json_path = out_dir / "no_scale_fp8_results.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        "# op12：无缩放因子 FP8 MMA 精度验证",
        "",
        f"- 上游固定提交：`{upstream_head}`",
        "- Gold：FlashInfer `tests/gdn/reference_delta_rule.py::blockwise_delta_rule`。",
        "- 输入：FlashInfer `tests/gdn/conftest.py::gen_qkv`，K 按上游测试先做 L2 normalize，再将 Q/K/V 直接转 E4M3。",
        "- 唯一数值变化：七组逻辑 GEMM 的输入使用 E4M3；中间量直接饱和转 E4M3；所有 MMA operand scale 固定为 1。",
        "- 保持不变：gate/beta、累计门控、三角逆、FP32 accumulator/epilogue、算法 QK scale=1/sqrt(128)。",
        "- 执行载体：unfused `torch._scaled_mm`；本结果验证精度，不声称上游 CuTe 融合 kernel 已改造完成。",
        "",
        "## 结论",
        "",
        f"- op12 阈值：{payload['summary']['op12_passed']}/{payload['summary']['total']} case 通过。",
        f"- 上游 FP16 严格 allclose：{payload['summary']['upstream_strict_passed']}/{payload['summary']['total']} case 通过。",
        "",
        "| case | layout | seq | heads Q/K/V | O rel-L2 | O cosine | state rel-L2 | scaled O rel-L2 | scaled state rel-L2 | zeroed nonzero max | op12 | strict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        rows.append(
            "| "
            + " | ".join(
                [
                    case.name,
                    case.layout,
                    ",".join(map(str, case.seq_lens)),
                    "/".join(map(str, case.heads)),
                    f"{case.no_scale.o_relative_l2:.6f}",
                    f"{case.no_scale.o_cosine:.6f}",
                    f"{case.no_scale.state_relative_l2:.6f}",
                    f"{case.tile_scaled.o_relative_l2:.6f}",
                    f"{case.tile_scaled.state_relative_l2:.6f}",
                    f"{case.max_zeroed_nonzero_rate:.6f}",
                    "PASS" if case.op12_pass else "FAIL",
                    "PASS" if case.upstream_strict_allclose else "FAIL",
                ]
            )
            + " |"
        )
    rows += [
        "",
        "## 判定规则",
        "",
        "- op12：O relative-L2≤0.20、O cosine≥0.98、final state relative-L2≤0.25、无 NaN/Inf。",
        "- 上游严格基线：O atol=2e-3/rtol=1e-3；state atol=1e-3/rtol=1e-4。",
        "- `scaled` 两列是同一算法、同一输入下的 tile amax/448 对照，只用于量化损失归因。",
    ]
    md_path = out_dir / "NO_SCALE_FP8_RESULTS.md"
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["summary"]["all_op12_passed"] else 2)


if __name__ == "__main__":
    main()

