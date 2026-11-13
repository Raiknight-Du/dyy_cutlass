#!/usr/bin/env python3
"""Generate the Chinese native CUDA GDN implementation/performance report."""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "OP12_NATIVE_CUDA_GDN_B200_IMPLEMENTATION_PERFORMANCE_ANALYSIS_CN.docx"
ASSETS = ROOT / "report_assets"
VALIDATION = ROOT / "validation"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "153B66"
MUTED = "667788"
LIGHT_GRAY = "F2F4F7"
LIGHT_BLUE = "EAF2FB"
LIGHT_GREEN = "EAF6EF"
LIGHT_ORANGE = "FFF3E5"
WHITE = "FFFFFF"
RED = "9B1C1C"


def set_run_font(run, size=None, bold=None, color=None, italic=None, mono=False):
    ascii_name = "Consolas" if mono else "Calibri"
    east_name = "Microsoft YaHei"
    run.font.name = ascii_name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), ascii_name)
    rfonts.set(qn("w:hAnsi"), ascii_name)
    rfonts.set(qn("w:eastAsia"), east_name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_paragraph_shading(paragraph, fill):
    ppr = paragraph._p.get_or_add_pPr()
    shd = ppr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        ppr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_shading(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcpr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tcpr = cell._tc.get_or_add_tcPr()
    tc_mar = tcpr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tcpr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    trpr.append(header)


def prevent_row_split(row):
    trpr = row._tr.get_or_add_trPr()
    trpr.append(OxmlElement("w:cantSplit"))


def set_table_geometry(table, widths):
    if sum(widths) != 9360:
        raise ValueError(f"table widths must sum to 9360 DXA, got {sum(widths)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tblpr = table._tbl.tblPr
    layout = tblpr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblpr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tblw = tblpr.find(qn("w:tblW"))
    tblw.set(qn("w:w"), "9360")
    tblw.set(qn("w:type"), "dxa")
    tblind = tblpr.find(qn("w:tblInd"))
    if tblind is None:
        tblind = OxmlElement("w:tblInd")
        tblpr.append(tblind)
    tblind.set(qn("w:w"), "120")
    tblind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        prevent_row_split(row)
        for index, cell in enumerate(row.cells):
            tcpr = cell._tc.get_or_add_tcPr()
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.append(tcw)
            tcw.set(qn("w:w"), str(widths[index]))
            tcw.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def write_cell(cell, text, bold=False, color=None, size=9.5, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold, color=color)


def add_table(doc, headers, rows, widths, aligns=None, font_size=9.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_shading(table.rows[0].cells[index], LIGHT_GRAY)
        write_cell(
            table.rows[0].cells[index],
            header,
            bold=True,
            color=NAVY,
            size=font_size,
            align=WD_ALIGN_PARAGRAPH.CENTER,
        )
    set_repeat_header(table.rows[0])
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            alignment = aligns[index] if aligns else WD_ALIGN_PARAGRAPH.LEFT
            write_cell(cells[index], value, size=font_size, align=alignment)
    set_table_geometry(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.10
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r2)
    else:
        run = p.add_run(text)
        set_run_font(run)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.167
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_callout(doc, title, text, fill=LIGHT_BLUE, color=NAVY):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.right_indent = Inches(0.12)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.12
    set_paragraph_shading(p, fill)
    run = p.add_run(f"{title}  ")
    set_run_font(run, bold=True, color=color)
    run = p.add_run(text)
    set_run_font(run, color=color)
    return p


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.0
    set_paragraph_shading(p, "F6F7F9")
    run = p.add_run(text)
    set_run_font(run, size=9.2, mono=True, color="23364A")


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = False
    run = p.add_run(text)
    set_run_font(run, size=9.5, italic=True, color=MUTED)


def add_figure(doc, filename, caption, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    picture = run.add_picture(str(ASSETS / filename), width=Inches(width))
    # Preserve a meaningful accessibility description in the generated DOCX.
    picture._inline.docPr.set("descr", caption)
    add_caption(doc, caption)


def page_break(doc):
    doc.add_page_break()


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    set_run_font(run, size=9, color=MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hr = hp.add_run("op12 | Native CUDA Gated DeltaNet Prefill on B200")
    set_run_font(hr, size=9, bold=True, color=MUTED)
    add_page_field(section.footer.paragraphs[0])


def read_summary():
    with (VALIDATION / "benchmark_native_summary.csv").open(encoding="utf-8-sig") as f:
        native = {row["name"]: row for row in csv.DictReader(f)}
    with (VALIDATION / "flashinfer_benchmark_summary.csv").open(encoding="utf-8-sig") as f:
        upstream = {row["name"]: row for row in csv.DictReader(f)}
    return native, upstream


def build():
    native, upstream = read_summary()
    doc = Document()
    configure_document(doc)
    doc.core_properties.title = "op12 原生 CUDA GDN Prefill B200 实现与性能分析"
    doc.core_properties.subject = "与固定 FlashInfer GDN 的功能差分、性能和 NCU/SASS 瓶颈分析"
    doc.core_properties.author = "Codex"
    doc.core_properties.keywords = "Gated DeltaNet, GDN, B200, CUDA, FlashInfer, NCU"

    # Cover / memo masthead
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("TECHNICAL FINDINGS")
    set_run_font(r, size=11, bold=True, color=BLUE)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("op12 原生 CUDA Gated DeltaNet Prefill")
    set_run_font(r, size=25, bold=True, color=NAVY)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(20)
    r = p.add_run("B200 功能等价实现、端到端性能与 NCU/SASS 瓶颈分析")
    set_run_font(r, size=15, color="3D5368")
    for label, value in (
        ("目标", "用独立 CUDA C 实现固定 FlashInfer GDN Prefill 的相同公开功能，并按相同方法测试性能"),
        ("对照", "FlashInfer commit e77a4a0d276367895c3b50a642fd8f326c03fb72"),
        ("硬件", "NVIDIA B200 / 148 SM / CUDA 13.0"),
        ("日期", "2026-08-29"),
        ("状态", "功能基线完成；CUTLASS/tcgen05 性能后端未实现"),
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(f"{label}：")
        set_run_font(r, bold=True, color=NAVY)
        r = p.add_run(value)
        set_run_font(r)
    add_callout(
        doc,
        "结论",
        "39/39 差分测试与两条 Compute Sanitizer 路径通过；40 个配置 x 3 次独立计时全部有限值。原生 CP 比原生非 CP 快 3.46-3.91 倍，但仍比固定 FlashInfer CP 慢 94.9-137.1 倍。主要原因是当前内核没有 tensor-core 指令，仍依赖标量 FP32 FFMA、shared-memory state 和逐 token 同步。",
        fill=LIGHT_BLUE,
    )
    page_break(doc)

    doc.add_heading("执行摘要与阅读边界", level=1)
    add_body(doc, "本报告回答两个问题：第一，独立 CUDA C 实现是否覆盖固定 FlashInfer GDN Prefill 的公开语义；第二，在相同 B200、数据生成、计时和 NCU 方法下，性能差距来自哪里。")
    add_bullet(doc, "“功能等价”指公开输入输出、状态递推和边界行为等价，不表示复制 FlashInfer 的内部 64-token 分块算法。")
    add_bullet(doc, "当前非 CP 是单 kernel 严格 token recurrence；CP 是 affine M/N、fixup、final prefill 三 kernel。")
    add_bullet(doc, "FlashInfer 固定实现使用 64-token chunk、七组逻辑 GEMM、三角逆、TMA/TMEM 和 tcgen05 MMA；当前原生 CUDA 没有这些性能机制。")
    add_bullet(doc, "FP8 实验不在本轮范围；Q/K/V 仅验证 FP16/BF16，state 覆盖 FP32/FP16/BF16/FP8 E4M3/E5M2 存储。")
    doc.add_heading("文档结构", level=2)
    add_table(
        doc,
        ["章节", "内容"],
        [
            ("1-2", "环境、GDN 递推语义和公开接口"),
            ("3", "非 CP 与 CP 三 kernel 实现和资源配置"),
            ("4", "差分正确性、容差和 sanitizer"),
            ("5", "端到端性能矩阵"),
            ("6", "NCU、SASS 和理论瓶颈归因"),
            ("7-附录", "优化优先级、复现命令和证据索引"),
        ],
        [1500, 7860],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
    )

    page_break(doc)
    doc.add_heading("1. 环境与源码固定", level=1)
    add_table(
        doc,
        ["项目", "固定值"],
        [
            ("GPU", "NVIDIA B200，148 SM，183359 MiB（191503007744 bytes）"),
            ("驱动 / CUDA", "580.159.03 / CUDA toolkit 13.0"),
            ("PyTorch", "2.11.0+cu130"),
            ("NCU", "Nsight Compute 2026.2.1.0"),
            ("FlashInfer", "e77a4a0d276367895c3b50a642fd8f326c03fb72；固定仓库 clean"),
            ("原生实现", "/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/native_cuda"),
            ("本地归档", str(ROOT)),
        ],
        [2100, 7260],
    )
    add_callout(doc, "隔离性", "原生扩展不链接或修改 FlashInfer kernel。FlashInfer 仅作为差分 oracle 和同方法性能对照。")
    doc.add_heading("2. GDN 计算语义与接口", level=1)
    doc.add_heading("2.1 逐 token 等价递推", level=2)
    add_body(doc, "实现内部将公开 [V,K] state 转为 [K,V] FP32 工作矩阵。对序列中的第 t 个 token，令 a=g[t,h]、b=beta[t,h]，执行：")
    add_code(
        doc,
        "S_decay = a * S_prev\n"
        "v_old  = K_t * S_decay\n"
        "delta  = b * (V_t - v_old)\n"
        "S_new  = S_decay + K_t^T * delta\n"
        "O_t    = scale * Q_t * S_new",
    )
    add_body(doc, "FlashInfer 通过 64-token blockwise 变换计算相同 delta rule；原生基线直接执行 recurrence。两者浮点运算顺序不同，因此差分验收使用 dtype 对应容差，而不是逐 bit 相等。")
    doc.add_heading("2.2 公开接口和能力", level=2)
    add_table(
        doc,
        ["维度", "原生实现", "与固定上游关系", "备注"],
        [
            ("布局", "TND [T,H,128] + cu_seqlens", "达成", "支持 varlen、尾块和零长度序列"),
            ("Q/K/V", "FP16 / BF16", "达成", "同 dtype，D 固定 128"),
            ("Head", "MHA / GQA / GVA", "达成", "Ho=max(Hq,Hv)，比例须整除"),
            ("State", "initial/final；5 种存储 dtype", "达成", "工作 state 始终 FP32"),
            ("Checkpoint", "64-token 整数倍", "达成", "非 CP 路径"),
            ("State pool", "int32 state_indices；in-place", "达成", "非 CP 路径"),
            ("use_cp", "False / True / auto", "达成", "CP 与上游相同：checkpoint/pool 组合被拒绝或 auto 回退"),
            ("QK L2 flag", "保留参数，固定 SM100 有效行为为关闭", "达成", "测试数据在 kernel 外归一化 K"),
        ],
        [1500, 2300, 1500, 4060],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        font_size=8.8,
    )
    add_figure(doc, "figure_1_architecture.png", "图 1  原生 CUDA 公开接口、dispatch 和三条执行路径（Draw.io 源文件同步保留）")
    doc.add_heading("3. 原生 CUDA 实现与硬件映射", level=1)
    doc.add_heading("3.1 Dispatch", level=2)
    add_body(doc, "非 CP grid 为 (num_seqs, Ho)，每个 CTA 独占一个 sequence/output-head。auto 条件与固定上游一致：num_seqs x Ho x 4 < SM 数时尝试 CP；B200 为 148 SM，因此 B1/H16 和 B2/H16 满足，B4/H16 不满足。")
    add_code(doc, "will_use_cp = (use_cp is True) or (use_cp == 'auto' and num_seqs * Ho * 4 < 148)")
    add_body(doc, "CP chunk 不固定为 4096。host 依据 SM/并行 head 自适应到 512-token 粒度；主测试均形成 8 partitions/head：S=8192/16384/32768 分别使用 1024/2048/4096 token partition。")

    doc.add_heading("3.2 非 CP：cuda_recurrent_v0", level=2)
    add_table(
        doc,
        ["项目", "配置", "职责 / 影响"],
        [
            ("Grid", "(B,Ho)", "B1/H16 仅 16 CTA，远小于 148 SM"),
            ("Block", "256 threads / 8 warps", "128 lanes 负责向量投影，其余线程并行更新 128x128 state"),
            ("State", "FP32 [K,V] in SMEM", "约 68.10 KiB/CTA；首尾才访问 GMEM"),
            ("寄存器", "70/thread（NCU）", "理论可 3 CTA/SM，但低 grid 下每个活跃 SM 仅 1 CTA"),
            ("同步", "每 token 多次 __syncthreads", "严格递推形成 barrier 等待"),
            ("指令", "FFMA/LDS/STS/BAR", "没有 Tensor Core 指令"),
        ],
        [1500, 2200, 5660],
    )

    page_break(doc)
    doc.add_heading("3.3 CP 三 kernel", level=2)
    add_body(doc, "CP 将一个 partition 的状态变化表示为内部 [K,V] 语义下的仿射变换 S_out=M·S_in+N。先并行求每个 partition 的 M/N，再修正边界，最后从正确边界独立生成输出。")
    add_table(
        doc,
        ["Kernel", "输入→输出", "Block / Grid（主配置）", "SMEM / reg", "主要限制"],
        [
            ("Affine M/N", "K,V,g,beta → FP32 M,N", "256 threads；(8,16)=128 CTA", "133.12 KiB；45 reg", "每 token 更新两个 128x128 矩阵；标量 FFMA"),
            ("Fixup", "M,N,initial → boundary/final state", "256 threads；(1,16)=16 CTA", "131.07 KiB；48 reg", "每 CTA 顺序扫描 8 partition；低 grid"),
            ("Final", "Q,K,V,boundary → O", "256 threads；(8,16)=128 CTA", "68.10 KiB；43 reg", "重复 token recurrence；标量 FFMA"),
        ],
        [1500, 2200, 2000, 1600, 2060],
        font_size=8.4,
    )
    add_body(doc, "与 FlashInfer CP 的内部差异：上游是 T precompute、M/N、fixup、final prefill 四 kernel，并复用 64-token T；原生实现没有独立 T，Affine 和 Final 各自执行 recurrence，因此只有三 kernel 但计算量更多。")

    doc.add_heading("3.4 理论工作量", level=2)
    add_body(doc, "非 CP 每 token/head 约含三组 128x128 FMA（K·S、Kᵀ·delta、Q·S），即约 6D² FLOP。CP Affine 还需维护 M/N，约 8D² FLOP/token/head；Final 再做约 6D²；Fixup 每 partition/head 额外做一个 D×D 乘 D×D 的 FP32 乘加。CP 用额外工作换取 partition 并行度。")
    doc.add_heading("4. 正确性与安全验证", level=1)
    doc.add_heading("4.1 测试数据与对比方法", level=2)
    add_body(doc, "每个测试使用固定 seed：Q/V 为正态分布，K 先在 FP32 做 D 维 L2 归一化再转换到 FP16/BF16，g/beta 为 FP32，initial state 为小幅 FP32 正态分布，scale=1/sqrt(128)。同一组 tensor 同时调用原生入口和固定 FlashInfer 入口。")
    add_table(
        doc,
        ["覆盖项", "样例", "验收"],
        [
            ("dtype/head/layout", "FP16/BF16；MHA/GQA/GVA；[1]、[63,65]、[64,127]", "输出与 final state assert_close"),
            ("CP", "[256]、[511,513]；两 dtype、三 head 模式", "与上游 use_cp=True 对比"),
            ("State dtype", "FP32/FP16/BF16/E4M3/E5M2", "输出和量化后 state 对比"),
            ("边界功能", "zero-length、pool、checkpoint、in-place", "有限值、状态位置和数值对比"),
        ],
        [1800, 3960, 3600],
    )
    doc.add_heading("4.2 容差", level=2)
    add_table(
        doc,
        ["路径", "Output", "Final state"],
        [
            ("FP16 非 CP", "atol=3e-3, rtol=2e-3", "atol=2e-3, rtol=2e-3"),
            ("BF16 非 CP", "atol=2e-2, rtol=2e-2", "atol=1e-2, rtol=5e-3"),
            ("FP16 CP", "atol=5e-3, rtol=4e-3", "atol=4e-3, rtol=4e-3"),
            ("BF16 CP", "atol=3e-2, rtol=3e-2", "atol=2e-2, rtol=2e-2"),
        ],
        [2200, 3580, 3580],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER] * 3,
    )
    add_callout(doc, "结果", "最终源码版本 39/39 tests passed（43 warnings 均来自上游 CuTe DSL 编译提示）；代表性非 CP 与 CP Compute Sanitizer memcheck 均为 0 errors。", fill=LIGHT_GREEN, color=DARK_BLUE)
    add_body(doc, "测试通过证明公开语义与固定实现落在约定容差内；不证明当前内部算法与上游相同，也不证明所有未枚举的超长输入具有相同误差分布。")
    page_break(doc)

    doc.add_heading("5. 端到端性能", level=1)
    doc.add_heading("5.1 方法", level=2)
    add_body(doc, "计时工具与上游审计保持一致：20 次 warmup、100 次 CUDA Event 样本、3 个独立进程，使用 rotating buffers；记录 median/P5/P95/mean/CV、温度、功耗、时钟和 GPU 进程。K 的 L2 归一化发生在计时区间外。")
    add_callout(doc, "规模", "40 configurations x 3 trials = 120 case results；全部 PASS、输出/state 有限；跨进程 median 的最大 CV 为 0.0154%。")
    add_figure(doc, "figure_2_performance.png", "图 2  B1/H16/D128/FP16 主配置端到端延迟（CUDA Event 三次独立进程中位数）")

    doc.add_heading("5.2 主配置结果", level=2)
    perf_rows = []
    for seq in (8192, 16384, 32768):
        nn = float(native[f"main_float16_b1_h16_s{seq}_False"]["median_of_medians_ms"])
        nc = float(native[f"main_float16_b1_h16_s{seq}_True"]["median_of_medians_ms"])
        fn = float(upstream[f"main_float16_b1_h16_s{seq}_False"]["median_of_medians_ms"])
        fc = float(upstream[f"main_float16_b1_h16_s{seq}_True"]["median_of_medians_ms"])
        perf_rows.append((str(seq), f"{nn:.4f}", f"{nc:.4f}", f"{fn:.6f}", f"{fc:.6f}", f"{nn/nc:.2f}x"))
    add_table(
        doc,
        ["S", "Native non-CP", "Native CP", "FI non-CP", "FI CP", "Native CP 加速"],
        perf_rows,
        [900, 1800, 1600, 1600, 1500, 1960],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER] * 6,
        font_size=8.9,
    )
    add_body(doc, "原生非 CP 随 S 近似线性增长。原生 CP 通过 128 CTA 覆盖更多 SM，获得 3.46-3.91 倍加速；但与 FlashInfer CP 的差距随 S 从 94.9 倍扩大到 137.1 倍，因为原生 Affine/Final 重复 recurrence，Fixup 还引入串行矩阵链。")

    doc.add_heading("5.3 并行度、head 与 state 消融", level=2)
    add_bullet(doc, "固定总 token=32768，非 CP 从 B1/H16/S32768 的 188.746 ms 降到 B8/H16/S4096 的 23.645 ms，再到 B16/H16/S2048 的 13.845 ms；grid 扩张是第一主因。")
    add_bullet(doc, "auto 在 B1/H16、B2/H16 选择 CP；从 B4/H16 起条件不满足并选择非 CP。")
    add_bullet(doc, "GQA 64/16/16 与 GVA 16/16/64 均约 47.60 ms；Ho=64 产生 64 CTA，但每 CTA 序列长度仍是 8192。")
    add_bullet(doc, "initial/final state 开关对 S8192 非 CP 的影响小于 0.03 ms，说明主耗时在递推而非 state 首尾搬运。")
    add_bullet(doc, "varlen [1024,3072,8192,20480] 为 117.84 ms，接近最长序列主导的 CTA 时间。")
    doc.add_heading("6. NCU、SASS 与瓶颈归因", level=1)
    doc.add_heading("6.1 NCU 配置", level=2)
    add_table(
        doc,
        ["ID", "配置", "路径", "核心 grid"],
        [
            ("N1", "B1/H16/S32768/D128/FP16", "non-CP", "16 CTA"),
            ("N2", "B8/H16/S4096/D128/FP16（总 token 相同）", "non-CP", "128 CTA"),
            ("N3", "B1/H16/S32768/D128/BF16", "non-CP", "16 CTA"),
            ("C1-C6", "B1/H16/S={8192,16384,32768}/{FP16,BF16}", "CP", "Affine/Final 128；Fixup 16"),
        ],
        [1000, 4700, 1500, 2160],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
    )
    add_body(doc, "每个 profile 使用 NCU full set、NVTX 仅捕获目标调用，并保留 .ncu-rep、details/raw CSV、SASS 和运行日志。NCU replay 的 kernel duration 用于阶段归因，不能替代真实 CUDA Event 端到端延迟。")

    doc.add_heading("6.2 非 CP：全卡并行度是第一瓶颈", level=2)
    add_table(
        doc,
        ["指标", "N1 欠并行 FP16", "N2 近满卡 FP16", "N3 欠并行 BF16"],
        [
            ("Duration", "188.06 ms", "23.64 ms", "192.56 ms"),
            ("Grid / waves/SM", "16 / 0.04", "128 / 0.29", "16 / 0.04"),
            ("SM throughput", "4.34%", "34.56%", "4.24%"),
            ("DRAM throughput", "0.04%", "0.29%", "0.04%"),
            ("L1/TEX throughput", "41.48%", "41.54%", "40.51%"),
            ("Eligible warps/scheduler", "0.34", "0.34", "0.34"),
            ("No eligible", "70.16%", "70.16%", "70.85%"),
            ("Tensor pipe", "0%", "0%", "0%"),
        ],
        [2300, 2350, 2350, 2360],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
        font_size=8.8,
    )
    add_body(doc, "N1→N2 在总 token 不变时提速约 7.96 倍，SM throughput 也约 8 倍增长，证明全卡 grid 是首要瓶颈。活跃 CTA 内的 eligible/no-eligible 基本不变，说明把卡铺满并未消除单 CTA 依赖链。")
    add_callout(doc, "Barrier 证据", "N1 平均每 warp 每条已发射指令间隔中有 2.27 cycles 等待 CTA barrier；NCU 规则将其归为平均 6.7-cycle issue 间隔的约 33.9%。")

    doc.add_heading("6.3 CP 三阶段", level=2)
    add_figure(doc, "figure_3_ncu_cp.png", "图 3  原生 CP 三阶段 NCU duration、资源和主要 stall 证据")
    cp_rows = []
    with (VALIDATION / "ncu_cp_stage_summary.csv").open(encoding="utf-8-sig") as f:
        cp = list(csv.DictReader(f))
    cp_lookup = {(r["profile"], r["kernel_short"]): r for r in cp}
    for seq in (8192, 16384, 32768):
        values = [cp_lookup[(f"native_cp_s{seq}_fp16", stage)] for stage in ("cp_1_affine_mn", "cp_2_fixup", "cp_3_final")]
        cp_rows.append((str(seq), *(f"{float(v['duration_ms']):.2f} ms ({float(v['duration_share_pct']):.1f}%)" for v in values)))
    add_table(
        doc,
        ["S", "Affine M/N", "Fixup", "Final"],
        cp_rows,
        [1200, 2720, 2720, 2720],
        aligns=[WD_ALIGN_PARAGRAPH.CENTER] * 4,
    )
    add_body(doc, "Affine 和 Final 随 token 数线性增长；Fixup 约 2.54 ms 基本不变，因为三种长度均固定 8 partitions/head。其低 grid 只有 16 CTA，SM throughput 约 2.47%，no-eligible 约 85.2%，是短序列时占比最大的不可并行开销。")
    add_body(doc, "FP16 与 BF16 三阶段 duration 差异小于约 0.4%，因为两者加载后都执行相同的 FP32 标量核心，没有 dtype 专属 Tensor Core 路径。")

    doc.add_heading("6.4 SASS 与理论归因", level=2)
    add_body(doc, "SASS 汇总只出现 FFMA、LDS、STS、BAR、LDG/STG 等族；没有 HMMA、UTCHMMA、UTCQMMA 或 tcgen05。S32768/FP16 的 Affine 执行约 1.342B 条 FFMA 和 2.148B 条 LDS；Final 约 805M FFMA 和 1.613B LDS。")
    add_table(
        doc,
        ["层级", "证据", "判定"],
        [
            ("全卡", "B1/H16 grid16；B8/H16 grid128 提速 7.96x", "低 BxHo 时欠并行"),
            ("活跃 SM", "0% Tensor pipe；0.34 eligible；70% no-eligible", "标量依赖链和同步限制 issue"),
            ("存储", "DRAM 0.04-0.30%；L1/TEX 40-52%", "不是 HBM 带宽瓶颈；主要为片上 state 流量"),
            ("资源", "68-133 KiB SMEM；43-70 reg/thread", "限制 CTA 资源，但当前 grid 通常先成为约束"),
            ("CP Fixup", "16 CTA、85% no-eligible、2.47% SM", "低 grid 串行 prefix 链"),
        ],
        [1500, 4400, 3460],
        font_size=8.8,
    )
    doc.add_heading("7. 结论与优化优先级", level=1)
    add_callout(doc, "功能结论", "原生 CUDA C 已覆盖固定 FlashInfer GDN Prefill 的主要公开功能，并通过 39 个差分测试与 sanitizer。", fill=LIGHT_GREEN, color=DARK_BLUE)
    add_callout(doc, "性能结论", "当前实现不是 FlashInfer 性能替代品。性能差距来自内部算法和硬件映射，而不是 FP16/BF16 或 state I/O：没有 64-token blockwise 七 GEMM、没有 tcgen05、没有 TMA/TMEM/warp specialization。", fill=LIGHT_ORANGE, color=RED)
    add_table(
        doc,
        ["优先级", "改造", "验收门禁"],
        [
            ("P0", "用 CUTLASS/CuTe 复现 64-token 七组 GEMM、三角逆和 FP32 epilogue；生成 tcgen05 MMA", "39 项差分保持；SASS 必须出现 tcgen05，Tensor pipe >0"),
            ("P1", "将 Q/K/V/state 搬运改为 TMA，多级 SMEM/TMEM pipeline，warp specialization", "消除 spill；量化 barrier/no-eligible 降幅"),
            ("P2", "CP 增加独立 T precompute，避免 Affine/Final 重复；Fixup 改树形/分层 scan", "S8192 Fixup 占比显著下降；CP 与 non-CP 一致"),
            ("P3", "按 BxHo、S、SMEM 自动选择 non-CP/CP 与 partition 数", "40-shape x3 trial 全量回归"),
        ],
        [900, 4860, 3600],
        font_size=8.7,
    )
    add_body(doc, "不建议继续只调当前 recurrence CTA 的线程数或 register 上限：NCU 已证明核心问题是标量算法、严格同步和低 grid；局部参数调优无法跨越 95-200 倍差距。")

    doc.add_heading("附录 A. 复现命令", level=1)
    add_code(doc, "# 差分正确性\npython -m pytest tests/test_correctness.py -q\n\n# 三次同方法计时\npython benchmarks/benchmark_native_gdn.py --suite all --implementation native --trial 1 --warmup 20 --iterations 100\n\n# NCU 完整矩阵\ntools/run_ncu_matrix.sh")
    add_body(doc, "实际运行需使用 op12/.venv、CUDA 13.0、TORCH_CUDA_ARCH_LIST=10.0a，并设置 FLASHINFER_REPO 指向固定提交。完整环境命令和原始采样见 validation 日志。")

    doc.add_heading("附录 B. 证据索引", level=1)
    add_table(
        doc,
        ["证据", "本地 / B200 位置", "用途"],
        [
            ("源码", "op12/native_cuda/csrc + op12_native_gdn", "独立 CUDA C 实现与 Python API"),
            ("正确性", "validation/pytest_correctness.xml + .log", "39/39 差分结果"),
            ("Sanitizer", "validation/compute_sanitizer_{noncp,cp}.log", "两条路径 0 errors"),
            ("计时", "validation/benchmark_native_trial{1,2,3}.jsonl", "120 个 case 原始 CUDA Event 样本"),
            ("计时汇总", "validation/benchmark_native_summary.csv；benchmark_native_vs_flashinfer.csv", "中位数、CV、速度比"),
            ("NCU 轻量", "validation/ncu/*.csv；ncu_*_summary.csv；sass_instruction_summary.csv", "指标和 SASS 归因"),
            ("NCU 完整", "/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/native_cuda/validation/ncu/*.ncu-rep", "B200 上可重新导入的 10 份报告"),
            ("完整性", "validation/ncu_reports.sha256；source_files.sha256", "NCU 和源码 SHA256"),
            ("绘图源", "report_assets/OP12_NATIVE_GDN_B200_DIAGRAMS.drawio", "三页 Draw.io 可编辑源文件"),
        ],
        [1500, 5760, 2100],
        font_size=8.2,
    )
    add_body(doc, "注：本地目录仅保留轻量 CSV/SASS 导出；约 1.8 GB 的完整 .ncu-rep 保留在 B200，并由 SHA256 清单索引。")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
