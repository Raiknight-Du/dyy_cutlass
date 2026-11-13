#!/usr/bin/env python3
"""Build deterministic report figures from native benchmark/NCU evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
OUT = ROOT / "report_assets"
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
)

NAVY = "#153B66"
BLUE = "#3478C0"
PALE_BLUE = "#EAF2FB"
GREEN = "#3E8E61"
PALE_GREEN = "#EAF6EF"
ORANGE = "#D9852B"
PALE_ORANGE = "#FFF3E5"
RED = "#B84B4B"
INK = "#152536"
MUTED = "#607286"
GRID = "#D9E1E8"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = list(FONT_CANDIDATES)
    if bold:
        candidates.insert(0, Path(r"C:\Windows\Fonts\msyhbd.ttc"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=INK, spacing=8):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rounded(draw, box, fill, outline, width=3, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill=NAVY, width=5):
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 17
    spread = 0.55
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=fill)


def build_architecture() -> None:
    image = Image.new("RGB", (1900, 1120), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 54), "op12 原生 CUDA GDN Prefill：功能等价与执行路径", font=font(48, True), fill=NAVY)
    draw.text((92, 120), "公开接口对齐固定 FlashInfer；内部采用独立 CUDA C 实现", font=font(25), fill=MUTED)

    input_box = (90, 205, 420, 410)
    rounded(draw, input_box, PALE_BLUE, BLUE)
    text_center(draw, input_box, "TND packed-varlen\nQ/K/V: FP16 或 BF16\ng/beta: FP32, D=128\ninitial/final state", font(27, True))

    diamond = [(650, 205), (800, 305), (650, 405), (500, 305)]
    draw.polygon(diamond, fill=PALE_ORANGE, outline=ORANGE)
    draw.line(diamond + [diamond[0]], fill=ORANGE, width=4)
    text_center(draw, (525, 235, 775, 375), "use_cp\nFalse / True / auto", font(28, True))
    arrow(draw, (420, 305), (500, 305))

    recurrent = (940, 175, 1775, 445)
    rounded(draw, recurrent, PALE_BLUE, BLUE)
    text_center(
        draw,
        recurrent,
        "非 CP：cuda_recurrent_v0（1 kernel）\n\n每个 (sequence, output-head) 一个 256-thread CTA\nFP32 [K,V] state 常驻 68.1 KiB SMEM\n逐 token：衰减 → K·S → delta → Kᵀ·delta → Q·S",
        font(25, True),
    )
    arrow(draw, (800, 265), (940, 265))

    cp_y = 570
    cp_boxes = [
        ((150, cp_y, 600, cp_y + 235), PALE_GREEN, GREEN, "① affine M/N\npartition-local\n内部 [K,V]：Sout = M·Sin + N\n128 CTA（主配置）"),
        ((725, cp_y, 1175, cp_y + 235), PALE_ORANGE, ORANGE, "② prefix fixup\n按 partition 顺序修正边界 state\n仅 16 CTA\nFP32 M·state + N"),
        ((1300, cp_y, 1750, cp_y + 235), PALE_GREEN, GREEN, "③ final prefill\n从正确边界 state 独立递推\n128 CTA（主配置）\n写 O 与 final state"),
    ]
    for box, fill, outline, label in cp_boxes:
        rounded(draw, box, fill, outline)
        text_center(draw, box, label, font(25, True))
    arrow(draw, (650, 405), (380, cp_y))
    arrow(draw, (600, cp_y + 117), (725, cp_y + 117))
    arrow(draw, (1175, cp_y + 117), (1300, cp_y + 117))

    note = (250, 905, 1650, 1045)
    rounded(draw, note, "#F6F7F9", GRID, width=2, radius=16)
    text_center(
        draw,
        note,
        "等价边界：输出、state、GQA/GVA、varlen、state dtype/checkpoint/pool 与上游做差分验证。\n性能边界：当前三条路径均为标量 FP32 FFMA；尚未复现 FlashInfer 的 64-token tcgen05 七组 GEMM。",
        font(24, True),
    )
    image.save(OUT / "figure_1_architecture.png", dpi=(180, 180))


def load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig") as stream:
        return {row["name"]: row for row in csv.DictReader(stream)}


def build_performance() -> None:
    native = load_csv(VALIDATION / "benchmark_native_summary.csv")
    upstream = load_csv(VALIDATION / "flashinfer_benchmark_summary.csv")
    sequences = (8192, 16384, 32768)
    series = (
        ("Native non-CP", BLUE, lambda s: native[f"main_float16_b1_h16_s{s}_False"]),
        ("Native CP", GREEN, lambda s: native[f"main_float16_b1_h16_s{s}_True"]),
        ("FlashInfer non-CP", ORANGE, lambda s: upstream[f"main_float16_b1_h16_s{s}_False"]),
        ("FlashInfer CP", RED, lambda s: upstream[f"main_float16_b1_h16_s{s}_True"]),
    )
    values = {
        label: [float(get(s)["median_of_medians_ms"]) for s in sequences]
        for label, _, get in series
    }

    image = Image.new("RGB", (1900, 1080), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "B200 端到端延迟：原生 CUDA 与固定 FlashInfer", font=font(46, True), fill=NAVY)
    draw.text((92, 110), "B=1, H=16, D=128, FP16；纵轴为对数尺度（越低越好）", font=font(24), fill=MUTED)
    plot = (170, 210, 1780, 865)
    left, top, right, bottom = plot
    ymin, ymax = 0.1, 300.0

    def ycoord(value: float) -> float:
        ratio = (math.log10(value) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
        return bottom - ratio * (bottom - top)

    for tick in (0.1, 0.3, 1, 3, 10, 30, 100, 300):
        y = ycoord(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{tick:g} ms"
        draw.text((left - 105, y - 15), label, font=font(20), fill=MUTED)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)

    group_width = (right - left) / len(sequences)
    bar_width = 76
    gap = 12
    for group, seq in enumerate(sequences):
        center = left + group_width * (group + 0.5)
        total_width = len(series) * bar_width + (len(series) - 1) * gap
        x0 = center - total_width / 2
        for index, (label, color, _) in enumerate(series):
            value = values[label][group]
            x = x0 + index * (bar_width + gap)
            y = ycoord(value)
            draw.rounded_rectangle((x, y, x + bar_width, bottom), radius=8, fill=color)
            text = f"{value:.3g}"
            bounds = draw.textbbox((0, 0), text, font=font(17, True))
            draw.text((x + (bar_width - (bounds[2] - bounds[0])) / 2, y - 28), text, font=font(17, True), fill=INK)
        seq_label = f"S={seq}"
        bounds = draw.textbbox((0, 0), seq_label, font=font(24, True))
        draw.text((center - (bounds[2] - bounds[0]) / 2, bottom + 24), seq_label, font=font(24, True), fill=INK)

    legend_x, legend_y = 220, 955
    for index, (label, color, _) in enumerate(series):
        x = legend_x + index * 390
        draw.rounded_rectangle((x, legend_y, x + 38, legend_y + 28), radius=5, fill=color)
        draw.text((x + 52, legend_y - 2), label, font=font(20, True), fill=INK)
    image.save(OUT / "figure_2_performance.png", dpi=(180, 180))


def build_ncu() -> None:
    with (VALIDATION / "ncu_cp_stage_summary.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    sequences = (8192, 16384, 32768)
    stages = (
        ("cp_1_affine_mn", "Affine M/N", BLUE),
        ("cp_2_fixup", "Fixup", ORANGE),
        ("cp_3_final", "Final", GREEN),
    )
    lookup = {(row["profile"], row["kernel_short"]): row for row in rows}

    image = Image.new("RGB", (1900, 1120), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "原生 CP 三阶段 NCU 证据链", font=font(46, True), fill=NAVY)
    draw.text((92, 110), "FP16，NCU replay 单 kernel 时长；不替代 CUDA Event 端到端延迟", font=font(24), fill=MUTED)

    left, top, right, bottom = 180, 230, 1060, 840
    max_total = 50.0
    for tick in (0, 10, 20, 30, 40, 50):
        y = bottom - tick / max_total * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 90, y - 14), f"{tick} ms", font=font(20), fill=MUTED)
    group_width = (right - left) / 3
    bar_width = 145
    for group, seq in enumerate(sequences):
        x = left + group_width * (group + 0.5) - bar_width / 2
        y_cursor = bottom
        for stage, label, color in stages:
            row = lookup[(f"native_cp_s{seq}_fp16", stage)]
            duration = float(row["duration_ms"])
            height = duration / max_total * (bottom - top)
            draw.rectangle((x, y_cursor - height, x + bar_width, y_cursor), fill=color)
            if height > 36:
                text_center(draw, (x, y_cursor - height, x + bar_width, y_cursor), f"{duration:.2f}", font(18, True), fill=WHITE)
            y_cursor -= height
        label = f"S={seq}"
        bounds = draw.textbbox((0, 0), label, font=font(23, True))
        draw.text((x + (bar_width - (bounds[2] - bounds[0])) / 2, bottom + 24), label, font=font(23, True), fill=INK)

    legend_y = 910
    for index, (_, label, color) in enumerate(stages):
        x = 210 + index * 280
        draw.rectangle((x, legend_y, x + 34, legend_y + 26), fill=color)
        draw.text((x + 46, legend_y - 3), label, font=font(20, True), fill=INK)

    evidence_boxes = [
        ((1170, 230, 1795, 405), PALE_BLUE, BLUE, "Affine M/N\n43.3% SM throughput\n0% Tensor pipe\n133.12 KiB SMEM / CTA"),
        ((1170, 470, 1795, 645), PALE_ORANGE, ORANGE, "Fixup（低 grid 串行链）\n约 2.54 ms，随 S 基本不变\n约 85.2% no-eligible\n仅 2.47% SM throughput"),
        ((1170, 710, 1795, 885), PALE_GREEN, GREEN, "Final prefill\n约 35.9% SM throughput\n0% Tensor pipe\n68.10 KiB SMEM / CTA"),
    ]
    for box, fill, outline, label in evidence_boxes:
        rounded(draw, box, fill, outline)
        text_center(draw, box, label, font(24, True))

    note = (230, 995, 1690, 1070)
    rounded(draw, note, "#F6F7F9", GRID, width=2, radius=14)
    text_center(draw, note, "结论：CP 解决全卡并行度，但没有解决活跃 SM 内的标量 FFMA、shared-memory 访问与同步等待。", font(23, True))
    image.save(OUT / "figure_3_ncu_cp.png", dpi=(180, 180))


def build_tensor_backend_architecture() -> None:
    image = Image.new("RGB", (1900, 980), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 42), "op12 Tensor Core 后端边界与调度", font=font(46, True), fill=NAVY)
    draw.text((82, 108), "同一 Python API；通过 OP12_NATIVE_BACKEND 选择实现，不能混淆独立性与性能结论", font=font(24), fill=MUTED)

    api = (610, 175, 1290, 300)
    rounded(draw, api, PALE_BLUE, BLUE)
    text_center(draw, api, "chunk_gated_delta_rule\nTND / FP16-BF16 / D128 / GQA-GVA / state", font(27, True))

    boxes = [
        ((80, 500, 560, 790), "scalar", PALE_ORANGE, ORANGE,
         "独立 CUDA C\n逐 token recurrence\nFP32 FFMA\n47.53 ms"),
        ((710, 500, 1190, 790), "wmma", PALE_GREEN, GREEN,
         "独立 CUDA C\n64-token 七组 GEMM\nHMMA.16816.F32\n15.39 ms"),
        ((1340, 500, 1820, 790), "tcgen05（默认）", PALE_BLUE, BLUE,
         "固定上游派生 CuTe DSL\nTMA + TMEM + UTCHMMA\n非 CP / CP 四 kernel\n0.258 ms"),
    ]
    centers = []
    for box, title, fill, outline, body in boxes:
        rounded(draw, box, fill, outline, width=4)
        draw.text((box[0] + 28, box[1] + 22), title, font=font(30, True), fill=outline)
        text_center(draw, (box[0] + 18, box[1] + 70, box[2] - 18, box[3] - 18), body, font(24, True))
        centers.append(((box[0] + box[2]) / 2, box[1]))
    for center in centers:
        arrow(draw, ((api[0] + api[2]) / 2, api[3]), center, width=5)

    note = (250, 855, 1650, 935)
    rounded(draw, note, "#F6F7F9", GRID, width=2, radius=14)
    text_center(draw, note, "验收口径：WMMA 证明独立 Tensor Core 数据流；性能一致由固定提交派生 tcgen05 路径实现，并明确披露复用关系。", font(23, True))
    image.save(OUT / "figure_4_tensor_backends.png", dpi=(180, 180))


def build_tcgen05_parity() -> None:
    with (VALIDATION / "tcgen05" / "benchmark_native_vs_flashinfer.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {row["name"]: row for row in rows}
    sequences = (8192, 16384, 32768)
    series = (
        ("non-CP", "False", BLUE),
        ("CP", "True", GREEN),
    )

    image = Image.new("RGB", (1900, 1050), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "tcgen05 派生后端与固定 FlashInfer 延迟比", font=font(44, True), fill=NAVY)
    draw.text((92, 108), "B1/H16/D128/FP16；40 配置 × 3 独立进程中的主配置，1.0 表示完全一致", font=font(24), fill=MUTED)

    left, top, right, bottom = 190, 235, 1740, 810
    lo, hi = 0.99, 1.055
    for tick in (0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05):
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID if tick != 1.0 else ORANGE, width=3 if tick == 1.0 else 2)
        draw.text((95, y - 14), f"{tick:.2f}x", font=font(20, tick == 1.0), fill=MUTED)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)

    group_width = (right - left) / len(sequences)
    bar_width, gap = 135, 30
    for gi, seq in enumerate(sequences):
        center = left + group_width * (gi + 0.5)
        for si, (label, path, color) in enumerate(series):
            row = lookup[f"main_float16_b1_h16_s{seq}_{path}"]
            ratio = float(row["native_over_flashinfer_latency_ratio"])
            x = center + (si - 0.5) * (bar_width + gap) - bar_width / 2
            y0 = bottom - (1.0 - lo) / (hi - lo) * (bottom - top)
            y = bottom - (ratio - lo) / (hi - lo) * (bottom - top)
            draw.rounded_rectangle((x, min(y, y0), x + bar_width, max(y, y0)), radius=8, fill=color)
            text = f"{ratio:.3f}x"
            bounds = draw.textbbox((0, 0), text, font=font(22, True))
            draw.text((x + (bar_width - (bounds[2] - bounds[0])) / 2, y - 34), text, font=font(22, True), fill=INK)
        seq_label = f"S={seq}"
        bounds = draw.textbbox((0, 0), seq_label, font=font(24, True))
        draw.text((center - (bounds[2] - bounds[0]) / 2, bottom + 25), seq_label, font=font(24, True), fill=INK)

    for i, (label, _, color) in enumerate(series):
        x = 610 + i * 330
        draw.rounded_rectangle((x, 900, x + 42, 930), radius=5, fill=color)
        draw.text((x + 56, 897), label, font=font(22, True), fill=INK)
    draw.text((530, 965), "全 40 配置：min 1.002x / median 1.017x / max 1.044x", font=font(25, True), fill=NAVY)
    image.save(OUT / "figure_5_tcgen05_parity.png", dpi=(180, 180))


def build_tensor_evidence() -> None:
    image = Image.new("RGB", (1900, 900), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((85, 45), "Tensor Core 指令与性能证据链", font=font(46, True), fill=NAVY)
    draw.text((87, 108), "B1/H16/S8192/D128/FP16 非 CP；Tensor pipe 为活跃 SM 内口径", font=font(24), fill=MUTED)
    cards = [
        ((70, 205, 580, 710), PALE_ORANGE, ORANGE, "标量 recurrence",
         ["指令：FFMA/LDS/STS/BAR", "Tensor pipe：0%", "延迟：47.53 ms", "性质：独立语义基线"]),
        ((695, 205, 1205, 710), PALE_GREEN, GREEN, "64-token WMMA",
         ["指令：HMMA.16816.F32", "Tensor pipe：2.34%", "延迟：15.39 ms", "性质：独立 CUDA C"]),
        ((1320, 205, 1830, 710), PALE_BLUE, BLUE, "SM100 tcgen05",
         ["指令：UTCHMMA + HMMA", "Tensor pipe：46.76%", "延迟：0.258 ms", "性质：固定上游派生"]),
    ]
    for box, fill, outline, title, lines in cards:
        rounded(draw, box, fill, outline, width=4)
        text_center(draw, (box[0] + 20, box[1] + 20, box[2] - 20, box[1] + 110), title, font(31, True), fill=outline)
        y = box[1] + 150
        for line in lines:
            draw.text((box[0] + 38, y), line, font=font(23, True), fill=INK)
            y += 78
    arrow(draw, (580, 455), (695, 455), fill=NAVY, width=5)
    arrow(draw, (1205, 455), (1320, 455), fill=NAVY, width=5)
    note = (250, 765, 1650, 845)
    rounded(draw, note, "#F6F7F9", GRID, width=2, radius=14)
    text_center(draw, note, "结论：HMMA 只解决矩阵乘指令；达到 FlashInfer 还必须具备 TMEM accumulator、TMA 多级流水、warp specialization 和分层三角逆。", font(23, True))
    image.save(OUT / "figure_6_tensor_evidence.png", dpi=(180, 180))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_architecture()
    build_performance()
    build_ncu()
    build_tensor_backend_architecture()
    build_tcgen05_parity()
    build_tensor_evidence()
    print(f"assets={OUT}")


if __name__ == "__main__":
    main()
