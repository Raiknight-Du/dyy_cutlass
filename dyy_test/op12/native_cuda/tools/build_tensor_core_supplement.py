#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

import build_word_report as br


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
OUT = ROOT / "OP12_GDN_B200_TENSOR_CORE_IMPLEMENTATION_SUPPLEMENT_CN.docx"


def load_csv(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def metric(rows, profile, kernel, name, default="-"):
    for row in rows:
        if row["profile"] == profile and row["kernel_short"] == kernel and row["metric"] == name:
            return f"{row['value']} {row['unit']}".strip()
    return default


def build() -> Path:
    parity = load_csv(VALIDATION / "tcgen05" / "benchmark_native_vs_flashinfer.csv")
    ncu_detail = load_csv(VALIDATION / "tcgen05" / "ncu_key_metrics.csv")
    ncu_raw = load_csv(VALIDATION / "tcgen05" / "ncu_raw_counters.csv")
    cp = load_csv(VALIDATION / "tcgen05" / "ncu_cp_stage_summary.csv")
    sass = load_csv(VALIDATION / "tcgen05" / "sass_instruction_summary.csv")
    parity_by_name = {row["name"]: row for row in parity}
    ratios = [float(row["native_over_flashinfer_latency_ratio"]) for row in parity]

    doc = Document()
    br.configure_document(doc)
    doc.core_properties.title = "op12 GDN B200 Tensor Core 实现补充与性能验收"
    doc.core_properties.subject = "独立 WMMA 与固定提交派生 tcgen05 后端边界、正确性、性能和 NCU/SASS 证据"
    doc.core_properties.author = "op12 engineering analysis"

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run("op12 | GDN Prefill on NVIDIA B200")
    br.set_run_font(r, size=10, bold=True, color=br.MUTED)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(38)
    r = p.add_run("TENSOR CORE SUPPLEMENT")
    br.set_run_font(r, size=12, bold=True, color=br.BLUE)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run("op12 GDN Tensor Core 后端补充实现")
    br.set_run_font(r, size=26, bold=True, color=br.NAVY)
    p = doc.add_paragraph()
    r = p.add_run("独立 WMMA 数据流、SM100 tcgen05 性能路径与 FlashInfer 同方法验收")
    br.set_run_font(r, size=16, color=br.DARK_BLUE)
    for label, value in (
        ("硬件", "NVIDIA B200 / SM100a / 148 SM / CUDA 13.0"),
        ("固定上游", "FlashInfer e77a4a0d276367895c3b50a642fd8f326c03fb72"),
        ("日期", "2026-08-29"),
        ("默认后端", "OP12_NATIVE_BACKEND=tcgen05"),
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        r = p.add_run(f"{label}：")
        br.set_run_font(r, size=11, bold=True, color=br.NAVY)
        r = p.add_run(value)
        br.set_run_font(r, size=11, color="152536")
    br.add_callout(
        doc,
        "结论",
        "性能一致目标已由固定提交派生 tcgen05 路径达到：40 配置延迟比 min/median/max="
        f"{min(ratios):.3f}/{statistics.median(ratios):.3f}/{max(ratios):.3f}。"
        "独立 CUDA WMMA 路径已具备七组 Tensor Core GEMM，但尚未达到 FlashInfer 性能；两者不能混写。",
        fill=br.LIGHT_BLUE,
        color=br.NAVY,
    )
    br.page_break(doc)

    doc.add_heading("执行摘要", level=1)
    br.add_bullet(doc, "新增独立 `cuda_blockwise_wmma_v1`：64-token chunk、七组 WMMA GEMM、FP32 累加/三角逆/state，SASS 出现 HMMA.16816.F32。")
    br.add_bullet(doc, "新增默认 `tcgen05` parity backend：验证并调用 op12 中固定 FlashInfer 源码，恢复 TMA、TMEM、12-warp specialization、UTCHMMA 和 CP 四 kernel。")
    br.add_bullet(doc, "两条新增路径均复用原公开 API；通过 OP12_NATIVE_BACKEND=tcgen05|wmma|scalar 切换。")
    br.add_bullet(doc, "39/39 功能测试通过；WMMA memcheck 0 errors；tcgen05 非 CP/CP memcheck 0 errors。")
    br.add_callout(doc, "边界", "tcgen05 性能路径是上游固定提交派生实现，不构成独立 CUDA C/CUTLASS C++ 重实现。若验收同时要求“独立实现”和“性能一致”，当前只完成前者的 WMMA 功能版与后者的派生性能版。", fill=br.LIGHT_ORANGE, color=br.RED)

    doc.add_heading("1. 后端边界与调度", level=1)
    br.add_figure(doc, "figure_4_tensor_backends.png", "图 1  scalar、独立 WMMA 与固定提交派生 tcgen05 后端的边界；可编辑 Draw.io 源文件同步保留", width=5.5)
    br.add_table(
        doc,
        ["后端", "独立性", "核心指令/机制", "用途"],
        [
            ("scalar", "独立 CUDA C", "FP32 FFMA/LDS/STS/BAR", "语义与边界基线"),
            ("wmma", "独立 CUDA C", "HMMA.16816.F32，FP32 accumulator", "64-token Tensor Core 开发线"),
            ("tcgen05", "固定上游派生", "UTCHMMA、TMEM、TMA、warp specialization", "默认性能路径"),
        ],
        [1400, 1700, 3600, 2660],
        font_size=9.2,
    )

    doc.add_heading("2. 独立 64-token WMMA 实现", level=1)
    br.add_body(doc, "一个 CTA 持有一个 sequence/output-head 的 FP32 [K,V] state，并按64 token推进。外部 Q/K/V 保持 FP16/BF16；每个 chunk 将 state 转为同输入 dtype，七组矩阵乘由8个 warp 按16x16 tile条带化执行。")
    br.add_table(
        doc,
        ["序号", "计算", "WMMA 输入", "FP32 后处理"],
        [
            ("1", "Wkk=K·Kᵀ", "row × col", "gate transfer、beta、严格下三角"),
            ("2", "Wqk=Q·Kᵀ", "row × col", "因果 mask、transfer、scale"),
            ("3/4", "KS/QS", "row × row state", "residual 与 inter-chunk O"),
            ("5", "NV=Ainv·R", "row × row", "Ainv 列乘 beta"),
            ("6", "Ointra=Wqkv·NV", "row × row", "与 QS 合成并写 O"),
            ("7", "dS=Kᵀ·delta", "col × row", "累加进衰减后的 FP32 state"),
        ],
        [800, 2200, 2300, 4060],
        font_size=8.8,
    )
    br.add_table(
        doc,
        ["资源", "WMMA v1", "影响"],
        [
            ("Block", "256 threads / 8 warps", "CTA 内同步式七 GEMM"),
            ("SMEM", "约208.25 KiB/CTA", "FP32 state + state input + Q/K/V + Ainv/QK/scratch"),
            ("指令", "HMMA.16816.F32 / BF16", "Tensor Core 已生效"),
            ("缺失", "TMEM/TMA/异步 pipeline", "仍有大量 shared 往返与 barrier"),
        ],
        [1500, 3000, 4860],
        font_size=9.1,
    )
    br.add_callout(doc, "正确性", "默认测试集切到 WMMA 后仍为 39/39 passed；代表性 Compute Sanitizer memcheck 为 0 errors。", fill=br.LIGHT_GREEN, color=br.DARK_BLUE)

    br.page_break(doc)
    doc.add_heading("3. SM100 tcgen05 性能路径", level=1)
    br.add_body(doc, "tcgen05 后端在首次解析时验证 FlashInfer 仓库位置、HEAD 和固定提交；随后直接调用归档 CuTe DSL kernel，避免每次调用执行 git 检查。它恢复上游真实硬件映射，而不是在 Python 层模拟性能。")
    br.add_table(
        doc,
        ["非 CP 项", "NCU 实测", "说明"],
        [
            ("Block / grid", "384 threads；B1/H16 grid16", "12 warp specialization；低 B×Ho 欠并行"),
            ("寄存器", "168/thread", "1 CTA/SM"),
            ("动态 SMEM", "226.30 KiB/CTA", "接近 SM100 单CTA上限"),
            ("SASS", "UTCHMMA + HMMA", "七组主 GEMM + 分层三角逆"),
            ("数据搬运", "TMA + TMEM", "state 与 accumulator 不再反复落 shared"),
        ],
        [1600, 2300, 5460],
        font_size=9.0,
    )
    br.add_figure(doc, "figure_6_tensor_evidence.png", "图 2  标量、WMMA 与 tcgen05 的指令和性能证据链")

    doc.add_heading("4. CP 四 kernel", level=1)
    br.add_body(doc, "默认 tcgen05 路径恢复固定 FlashInfer 的四阶段 CP：T预计算、M/N预计算、跨partition fixup、final prefill。下表为B1/H16/FP16的NCU replay单kernel时长；不能替代真实CUDA Event端到端延迟。")
    stage_labels = {
        "cp_0_t_precompute": "T precompute",
        "cp_1_mn_precompute": "M/N precompute",
        "cp_2_fixup": "Fixup",
        "cp_3_final": "Final prefill",
    }
    cp_rows = []
    for seq in (8192, 16384, 32768):
        profile = f"tcgen05_cp_s{seq}_fp16"
        for row in cp:
            if row["profile"] == profile:
                cp_rows.append((
                    str(seq),
                    stage_labels[row["kernel_short"]],
                    row["grid_size"],
                    row["block_size"],
                    f"{float(row['duration_ms']) * 1000:.2f} us",
                    f"{float(row['duration_share_pct']):.1f}%",
                ))
    # Keep the CP stage table with its body rows; an orphaned header is hard to read.
    br.page_break(doc)
    br.add_table(doc, ["S", "阶段", "Grid", "Block", "NCU duration", "占比"], cp_rows, [700, 1900, 1600, 1500, 2200, 1460], font_size=8.1)
    br.add_body(doc, "T阶段随64-token块数增长；M/N和Final随partition工作增长；Fixup保持grid32且约28-29 us，因此其占比从S8192的19.5%降到S32768的7.7%。")

    br.page_break(doc)
    doc.add_heading("5. 同方法性能验收", level=1)
    br.add_body(doc, "数据生成、rotating buffer、20次warmup、100个CUDA Event样本和3个独立进程均沿用上一版。Q/V正态分布，K在计时区间外做L2归一化，gate/beta为FP32，initial/final state开启，scale=1/sqrt(128)。")
    br.add_figure(doc, "figure_5_tcgen05_parity.png", "图 3  tcgen05 派生后端与固定 FlashInfer 主配置延迟比")
    perf_rows = []
    for seq in (8192, 16384, 32768):
        for path, label in (("False", "non-CP"), ("True", "CP")):
            row = parity_by_name[f"main_float16_b1_h16_s{seq}_{path}"]
            perf_rows.append((
                str(seq), label,
                f"{float(row['native_ms']):.6f}",
                f"{float(row['flashinfer_ms']):.6f}",
                f"{float(row['native_over_flashinfer_latency_ratio']):.3f}x",
            ))
    br.add_table(doc, ["S", "路径", "tcgen05派生 ms", "FlashInfer ms", "延迟比"], perf_rows, [850, 1300, 2500, 2500, 2210], font_size=9.0)
    br.add_callout(doc, "验收", f"40/40配置均PASS且输出/state有限；延迟比min={min(ratios):.3f}x、median={statistics.median(ratios):.3f}x、max={max(ratios):.3f}x，满足≤1.05x的同量级门槛。", fill=br.LIGHT_GREEN, color=br.DARK_BLUE)

    doc.add_heading("6. NCU 与 SASS 证据", level=1)
    noncp_profiles = (
        ("tcgen05_noncp_underfilled_fp16", "B1/H16/S32768 FP16"),
        ("tcgen05_noncp_nearfull_fp16", "B8/H16/S4096 FP16"),
        ("tcgen05_noncp_underfilled_bf16", "B1/H16/S32768 BF16"),
    )
    ncu_rows = []
    for profile, label in noncp_profiles:
        ncu_rows.append((
            label,
            metric(ncu_detail, profile, "recurrent_tcgen05", "Duration"),
            metric(ncu_detail, profile, "recurrent_tcgen05", "Waves Per SM"),
            metric(ncu_raw, profile, "recurrent_tcgen05", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            metric(ncu_raw, profile, "recurrent_tcgen05", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
            metric(ncu_detail, profile, "recurrent_tcgen05", "No Eligible"),
        ))
    br.add_table(doc, ["配置", "Duration", "Waves/SM", "SM throughput", "Tensor elapsed", "No eligible"], ncu_rows, [1900, 1350, 1000, 1800, 1750, 1560], font_size=8.2)
    opcode_totals = defaultdict(int)
    for row in sass:
        opcode = row["opcode"].upper()
        if "UTCHMMA" in opcode or opcode.startswith("HMMA"):
            opcode_totals[(row["kernel_short"], row["opcode"])] += int(row["instructions_executed"])
    opcode_rows = sorted(opcode_totals.items(), key=lambda item: (item[0][0], item[0][1]))
    br.add_table(doc, ["Kernel", "Tensor opcode", "9配置累计执行数"], [(k, op, f"{count:,}") for ((k, op), count) in opcode_rows], [2800, 3300, 3260], font_size=8.2)
    br.add_body(doc, "WMMA快速NCU的活跃SM Tensor pipe为2.34%，tcgen05同口径为46.76%。这证明性能差距来自整体硬件映射，而不只是是否出现一条Tensor Core指令。")

    doc.add_heading("7. 安全、限制与结论", level=1)
    br.add_table(
        doc,
        ["检查", "结果", "说明"],
        [
            ("WMMA memcheck", "0 errors", "独立非CP代表性GQA/varlen"),
            ("tcgen05非CP memcheck", "0 errors", "固定CUTE_DSL_ARCH=sm_100a"),
            ("tcgen05 CP memcheck", "0 errors", "关闭cuGetProcAddress API-error报告，仅检查内存错误"),
            ("差分测试", "39/39", "FP16/BF16、MHA/GQA/GVA、state/checkpoint/pool/CP"),
        ],
        [2200, 1600, 5560],
        font_size=9.0,
    )
    br.add_callout(doc, "最终判定", "“运行性能与FlashInfer一致”已经实现；“完全独立CUDA/CUTLASS C++且性能一致”尚未实现。下一阶段若继续独立化，必须把上游CuTe DSL的tcgen05 atom、TMEM layout、TMA descriptor、12-warp pipeline和分层三角逆逐项移植，不能把WMMA版本的15.39 ms称为性能完成。", fill=br.LIGHT_ORANGE, color=br.RED)

    doc.add_heading("附录 A. 复现", level=1)
    br.add_code(doc, "# 三后端切换\nOP12_NATIVE_BACKEND=tcgen05 python ...\nOP12_NATIVE_BACKEND=wmma python ...\nOP12_NATIVE_BACKEND=scalar python ...\n\n# 完整正确性与三轮性能\ntools/run_tcgen05_validation.sh\n\n# 9配置 NCU full set\ntools/run_ncu_tcgen05_matrix.sh")
    doc.add_heading("附录 B. 证据索引", level=1)
    br.add_table(
        doc,
        ["证据", "位置"],
        [
            ("WMMA源码", "csrc/gdn_prefill_wmma.cu"),
            ("tcgen05固定检查", "op12_native_gdn/tcgen05_backend.py"),
            ("39项测试", "validation/tcgen05/pytest_tcgen05.log"),
            ("三轮原始计时", "validation/tcgen05/benchmark_tcgen05_trial{1,2,3}.jsonl"),
            ("性能对比", "validation/tcgen05/benchmark_native_vs_flashinfer.csv"),
            ("NCU/SASS汇总", "validation/tcgen05/ncu_* + sass_instruction_summary.csv"),
            ("完整NCU", "/home/ubuntu/.../op12/native_cuda/validation/ncu_tcgen05/*.ncu-rep"),
            ("Draw.io", "report_assets/OP12_TENSOR_CORE_BACKENDS.drawio"),
        ],
        [2500, 6860],
        font_size=8.8,
    )

    doc.save(OUT)
    print(OUT)
    return OUT


if __name__ == "__main__":
    build()
