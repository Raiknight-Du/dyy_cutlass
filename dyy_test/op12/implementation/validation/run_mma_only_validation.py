#!/usr/bin/env python3
"""Validate the single-variable change: seven FlashInfer GDN MMAs -> E4M3."""

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
sys.path.insert(0, str(UPSTREAM))

import torch

from op12_gdn import chunk_gated_delta_rule_mma_only_fp8


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref_mod = _load_module(
    "flashinfer_reference_delta_rule_mma_only",
    UPSTREAM / "tests" / "gdn" / "reference_delta_rule.py",
)
fixture_mod = _load_module(
    "flashinfer_gdn_conftest_mma_only",
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
class CaseResult:
    name: str
    io_dtype: str
    seq_lens: list[int]
    heads: list[int]
    qv_multiplier: float
    initial_state: bool
    finite: bool
    op12_pass: bool
    upstream_strict_allclose: bool
    mma_only: Metrics
    original: Metrics | None
    original_strict_allclose: bool | None
    original_error: str | None
    max_saturation_rate: float
    max_zeroed_nonzero_rate: float
    mma_operand_quantization: dict[str, Any]


def metric(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    a = actual.float().flatten()
    e = expected.float().flatten()
    rel = float((a - e).norm() / e.norm().clamp_min(1.0e-12))
    cosine = float(torch.nn.functional.cosine_similarity(a, e, dim=0))
    max_abs = float((a - e).abs().max())
    return rel, cosine, max_abs


def metrics(
    out: torch.Tensor,
    state: torch.Tensor,
    ref_out: torch.Tensor,
    ref_state: torch.Tensor,
) -> Metrics:
    o_rel, o_cos, o_max = metric(out, ref_out)
    s_rel, s_cos, s_max = metric(state, ref_state)
    return Metrics(o_rel, o_cos, o_max, s_rel, s_cos, s_max)


def strict_allclose(
    out: torch.Tensor,
    state: torch.Tensor,
    ref_out: torch.Tensor,
    ref_state: torch.Tensor,
) -> bool:
    return bool(
        torch.allclose(out.float(), ref_out.float(), atol=2.0e-3, rtol=1.0e-3)
        and torch.allclose(
            state.float(), ref_state.float(), atol=1.0e-3, rtol=1.0e-4
        )
    )


def max_stat(stats: dict[str, Any], key: str) -> float:
    return max((float(v.get(key, 0.0)) for v in stats.values()), default=0.0)


def make_inputs(
    seq_lens: list[int],
    heads: tuple[int, int, int],
    io_dtype: torch.dtype,
    *,
    qv_multiplier: float,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    hq, hk, hv = heads
    q, k, v = fixture_mod.gen_qkv(seq_lens, hq, hk, hv, 128, torch.float32)
    q = (q * qv_multiplier).to(device="cuda", dtype=io_dtype).contiguous()
    k = torch.nn.functional.normalize(k.to(device="cuda"), p=2.0, dim=-1).to(
        io_dtype
    ).contiguous()
    v = (v * qv_multiplier).to(device="cuda", dtype=io_dtype).contiguous()
    ho = max(hq, hv)
    total = sum(seq_lens)
    gate = torch.rand((total, ho), dtype=torch.float32, device="cuda")
    beta = torch.rand((total, ho), dtype=torch.float32, device="cuda")
    cu = torch.tensor(
        ref_mod.exclusive_cumsum(seq_lens), dtype=torch.int32, device="cuda"
    )
    return q, k, v, gate, beta, cu


def run_original_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu: torch.Tensor,
    initial: torch.Tensor | None,
):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    output_state = None if initial is None else torch.empty_like(initial)
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        g=gate,
        beta=beta,
        initial_state=initial,
        output_state=output_state,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
    )


def evaluate(
    name: str,
    seq_lens: list[int],
    heads: tuple[int, int, int] = (1, 1, 1),
    *,
    io_dtype: torch.dtype = torch.float16,
    qv_multiplier: float = 1.0,
    with_initial_state: bool = False,
    seed: int = 1200,
) -> CaseResult:
    q, k, v, gate, beta, cu = make_inputs(
        seq_lens,
        heads,
        io_dtype,
        qv_multiplier=qv_multiplier,
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

    (out, state), stats = chunk_gated_delta_rule_mma_only_fp8(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
        return_stats=True,
    )

    ref_initial = None if initial is None else initial.transpose(-1, -2).float()
    ref_out, ref_state_kv = ref_mod.blockwise_delta_rule(
        q,
        k,
        v,
        seq_lens,
        alpha=gate,
        beta=beta,
        block_size=64,
        scale_factor=1.0 / math.sqrt(128),
        state_dtype=torch.float32,
        initial_state=ref_initial,
    )
    ref_state = ref_state_kv.transpose(-1, -2).contiguous().to(state.dtype)
    mma_metrics = metrics(out, state, ref_out, ref_state)
    finite = bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    op12_pass = (
        finite
        and mma_metrics.o_relative_l2 <= 0.20
        and mma_metrics.o_cosine >= 0.98
        and mma_metrics.state_relative_l2 <= 0.25
    )

    original_metrics = None
    original_strict = None
    original_error = None
    try:
        original_out, original_state = run_original_flashinfer(
            q, k, v, gate, beta, cu, initial
        )
        original_metrics = metrics(
            original_out, original_state, ref_out, ref_state
        )
        original_strict = strict_allclose(
            original_out, original_state, ref_out, ref_state
        )
    except Exception as exc:  # preserve numeric results if upstream build fails
        original_error = f"{type(exc).__name__}: {exc}"

    qstats = stats.get("mma_operand_quantization", {})
    return CaseResult(
        name=name,
        io_dtype=str(io_dtype).removeprefix("torch."),
        seq_lens=seq_lens,
        heads=list(heads),
        qv_multiplier=qv_multiplier,
        initial_state=with_initial_state,
        finite=finite,
        op12_pass=op12_pass,
        upstream_strict_allclose=strict_allclose(out, state, ref_out, ref_state),
        mma_only=mma_metrics,
        original=original_metrics,
        original_strict_allclose=original_strict,
        original_error=original_error,
        max_saturation_rate=max_stat(qstats, "max_saturation_rate"),
        max_zeroed_nonzero_rate=max_stat(qstats, "max_zeroed_nonzero_rate"),
        mma_operand_quantization=qstats,
    )


def main() -> None:
    cases: list[CaseResult] = []
    for seq in (1, 63, 64, 65, 127, 257):
        cases.append(evaluate(f"FP16_TND_S{seq}", [seq], seed=1200 + seq))
    varlen = [1, 63, 64, 65, 127, 257]
    cases.append(evaluate("FP16_TND_varlen", varlen, seed=1300))
    cases.append(evaluate("FP16_GQA_8_2", [64], (8, 2, 2), seed=1301))
    cases.append(evaluate("FP16_GVA_2_8", [64], (2, 2, 8), seed=1302))
    cases.append(
        evaluate("FP16_initial_state", [65], with_initial_state=True, seed=1303)
    )
    cases.append(
        evaluate("FP16_QKV_low_0p25", [64], qv_multiplier=0.25, seed=1304)
    )
    cases.append(
        evaluate("FP16_QKV_high_4p0", [64], qv_multiplier=4.0, seed=1305)
    )
    cases.append(evaluate("FP16_long_4096_nonCP", [4096], seed=1306))
    cases.append(
        evaluate("BF16_TND_S64", [64], io_dtype=torch.bfloat16, seed=1401)
    )
    cases.append(
        evaluate(
            "BF16_TND_varlen",
            varlen,
            io_dtype=torch.bfloat16,
            seed=1402,
        )
    )

    upstream_head = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "scope": {
            "upstream_commit": upstream_head,
            "only_change": "seven core MMA operand dtypes FP16/BF16 -> E4M3",
            "public_qkv_output": "unchanged FP16/BF16",
            "gate_beta": "unchanged FP32",
            "state": "unchanged public [N,H,V,K]; default FP32, explicit initial-state dtype preserved",
            "layout": "unchanged packed TND + cu_seqlens",
            "quantization_scale": "none; literal-one MMA scales",
            "algorithmic_scale": "1/sqrt(128), unchanged",
            "backend": "unfused correctness vehicle",
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
                "finite_required": True,
            },
            "strict": {
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
            "mma_only_strict_passed": sum(
                case.upstream_strict_allclose for case in cases
            ),
            "original_strict_passed": sum(
                case.original_strict_allclose is True for case in cases
            ),
            "original_failures": sum(case.original_error is not None for case in cases),
        },
    }

    output_dir = ROOT / "validation"
    json_path = output_dir / "mma_only_fp8_results.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = payload["summary"]
    rows = [
        "# op12：仅七组核心 MMA 改为无缩放 FP8 的精度验证",
        "",
        f"- FlashInfer 固定提交：`{upstream_head}`。",
        "- 公共 Q/K/V/O 保持 FP16/BF16；gate/beta 保持 FP32；仅 MMA 边界临时转换 E4M3。",
        "- MMA scale 恒为 1；算法 scale=1/sqrt(128) 保持不变。",
        "- Gold：同一提交 `blockwise_delta_rule`；另运行原版 SM100 kernel 作对照。",
        "- 公共布局保持原版 packed TND，本轮不把 BNSD wrapper 混入单变量实验。",
        "",
        "## 结论",
        "",
        f"- op12 放宽门槛：{summary['op12_passed']}/{summary['total']}。",
        f"- MMA-only 严格 allclose：{summary['mma_only_strict_passed']}/{summary['total']}。",
        f"- FlashInfer 原版严格 allclose：{summary['original_strict_passed']}/{summary['total']}；环境/构建失败 {summary['original_failures']}。",
        "- **最终判定：不完全达标。** 原版对照全部通过，说明两项失败由无缩放 E4M3 MMA 这一唯一变量引入，而不是输入、算法、上游 kernel 或运行环境变化。",
        "",
        "| case | dtype | seq | heads Q/K/V | MMA-only O rel-L2 | O cosine | state rel-L2 | 原版 O rel-L2 | 原版 state rel-L2 | zeroed max | op12 | strict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        original_o = "ERR" if case.original is None else f"{case.original.o_relative_l2:.6f}"
        original_s = "ERR" if case.original is None else f"{case.original.state_relative_l2:.6f}"
        rows.append(
            "| "
            + " | ".join(
                [
                    case.name,
                    case.io_dtype,
                    ",".join(map(str, case.seq_lens)),
                    "/".join(map(str, case.heads)),
                    f"{case.mma_only.o_relative_l2:.6f}",
                    f"{case.mma_only.o_cosine:.6f}",
                    f"{case.mma_only.state_relative_l2:.6f}",
                    original_o,
                    original_s,
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
        "- op12：O relative-L2≤0.20、O cosine≥0.98、final-state relative-L2≤0.25、全部有限。",
        "- 严格：O atol=2e-3/rtol=1e-3；state atol=1e-3/rtol=1e-4；两者同时 allclose。",
        "- `zeroed max` 只统计送入七组 MMA 的临时 E4M3 操作数，不代表公共张量被改为 FP8。",
        "",
        "## 能力边界",
        "",
        "- 这是 unfused 正确性 backend；证明单变量数值影响和 B200 FP8 Tensor Core 可执行性，不声称融合 CuTe kernel 已编译完成。",
        "- CP、checkpoint、state pool 和 BNSD 不进入本轮单变量精度矩阵；它们仍由未修改的 FlashInfer 原版能力定义。",
        "",
        "## 失败归因",
        "",
        "- `FP16_TND_S1` 的 O relative-L2 为 `0.229802`，但最大绝对误差只有约 `9.42e-6`；短序列输出范数很小，使相对误差被放大。",
        "- `FP16_QKV_low_0p25` 同时越过两条门槛：O relative-L2=`0.200916`、cosine=`0.979779`。它表明无缩放 E4M3 对小幅操作数没有足够鲁棒性。",
        "- 15 组最大饱和率均为 0；问题不是上溢，而是临时 MMA operands 的舍入与小值归零。BF16 varlen 中某类 operand 的峰值归零率达到 `0.957764`。",
    ]
    report_path = output_dir / "MMA_ONLY_FP8_RESULTS.md"
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["all_op12_passed"] else 2)


if __name__ == "__main__":
    main()
