#!/usr/bin/env python3

import csv
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2_eq_corrected"
RUN = OP2 / "results/standard_search_20260727"
SHAPES = OP2 / "equivalent_shapes.csv"
BASELINE = RUN / "op2_eq_corrected_baseline_summary.csv"
BEST = RUN / "op2_eq_corrected_best_final_summary.csv"
BEST_RAW = RUN / "op2_eq_corrected_best_final_raw.csv"
CONFIGS = OP2 / "results/op2_eq_corrected_best_configs.csv"
LEGAL = (
    ROOT
    / "dyy_test/op2_eq/search_space/legacy_legal_scan/kernel_configs.csv"
)
COMPARISON = OP2 / "results/op2_eq_corrected_baseline_vs_best.csv"
REPORT = OP2 / "ANALYSIS_CN.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def config_id(variant: str) -> str:
    return variant.rsplit("_C", 1)[0]


def main() -> None:
    shape_rows = {int(row["shape"]): row for row in read_csv(SHAPES)}
    baseline = {int(row["shape"]): row for row in read_csv(BASELINE)}
    best = {int(row["shape"]): row for row in read_csv(BEST)}
    configs = {int(row["shape"]): row for row in read_csv(CONFIGS)}
    legal = {row["kernel_id"]: row for row in read_csv(LEGAL)}
    expected = list(range(1, 14))
    for name, values in (
        ("shape", shape_rows),
        ("baseline", baseline),
        ("best", best),
        ("config", configs),
    ):
        if sorted(values) != expected:
            raise RuntimeError(f"{name}: expected complete 13-shape input")

    rows: list[dict[str, object]] = []
    for shape in expected:
        dimensions = shape_rows[shape]
        base = baseline[shape]
        winner = best[shape]
        config = configs[shape]
        resource = legal[config_id(config["variant"])]
        base_ms = float(base["mean_ms"])
        best_ms = float(winner["mean_ms"])
        rows.append(
            {
                "shape": shape,
                "dense_m": dimensions["dense_m"],
                "dense_n": dimensions["dense_n"],
                "dense_k": dimensions["dense_k"],
                "sparse_m": dimensions["sparse_m"],
                "sparse_n": dimensions["sparse_n"],
                "sparse_k": dimensions["sparse_k"],
                "op2_m": dimensions["op2_m"],
                "op2_n": dimensions["op2_n"],
                "op2_k": dimensions["op2_k"],
                "flop_difference_pct": dimensions["flop_difference_pct"],
                "byte_difference_pct": dimensions["byte_difference_pct"],
                "baseline_variant": base["variant"],
                "baseline_mean_ms": f"{base_ms:.9f}",
                "baseline_stdev_ms": base["stdev_ms"],
                "best_variant": config["variant"],
                "best_mean_ms": f"{best_ms:.9f}",
                "best_stdev_ms": winner["stdev_ms"],
                "best_ci95_half_ms": winner["ci95_half_ms"],
                "improvement_pct": (
                    f"{(base_ms - best_ms) / base_ms * 100:.4f}"
                ),
                "selection_stage": config["selection_stage"],
                "selection_samples": config["selection_samples"],
                "sm_mode": config["sm_mode"],
                "tile_m": config["tile_m"],
                "tile_n": config["tile_n"],
                "tile_k": config["tile_k"],
                "cluster": (
                    f"{config['cluster_m']}x{config['cluster_n']}x"
                    f"{config['cluster_k']}"
                ),
                "epilogue": (
                    "auto"
                    if config["epilogue_auto"].lower() == "true"
                    else f"{config['epilogue_m']}x{config['epilogue_n']}"
                ),
                "stages": resource["stages"],
                "kernel_smem_kib": (
                    f"{int(resource['kernel_smem_bytes']) / 1024:.3f}"
                ),
            }
        )

    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_baseline = sum(float(row["baseline_mean_ms"]) for row in rows)
    total_best = sum(float(row["best_mean_ms"]) for row in rows)
    total_gain = (total_baseline - total_best) / total_baseline * 100
    gpu_rows = read_csv(BEST_RAW)
    if len(gpu_rows) != 325:
        raise RuntimeError(f"Expected 325 final samples, got {len(gpu_rows)}")
    pstates = sorted({row["pstate"] for row in gpu_rows})
    clocks = [int(float(row["idle_sm_mhz"])) for row in gpu_rows]
    temperatures = [int(float(row["temperature_c"])) for row in gpu_rows]

    lines = [
        "# OP2 修正等价 Shape 重测报告",
        "",
        "## 结论",
        "",
        (
            f"- 修正后的 13-shape pure-matmul baseline 总时延为 "
            f"`{total_baseline:.6f} ms`。"
        ),
        (
            f"- 每个 shape 扫描 1,080 个合法配置并完成独立复测后，"
            f"best 总时延为 `{total_best:.6f} ms`，整体改善 "
            f"`{total_gain:.2f}%`。"
        ),
        "- 最终结果使用固定随机输入，不使用全 0、全 1 或现场重新生成的数据。",
        "",
        "## 等价口径",
        "",
        "- 稠密参考理论字节：`Qd = 0.5625*Md*Kd + 0.5625*Nd*Kd + 2*Md*Nd`。",
        "- 当前 OP2 稀疏理论字节：`Qs = 0.53125*Ms*Ks + 0.34375*Ns*Ks + 2*Ms*Ns`。",
        (
            "- OP2 的 `0.34375 Byte/逻辑权重元素` 已包含压缩 NVFP4 值、"
            "pair-wise 4:8 metadata 与当前 SFB 布局的 scale；"
            "`0.53125` 包含稠密 NVFP4 值与当前 SFA scale。"
        ),
        "- 等计算量约束：`Ms*Ns*Ks = 2*Md*Nd*Kd`，离散对齐后误差见结果表。",
        (
            "- 用户逻辑矩阵为 `D[M_s,N_s] = A_dense[M_s,K_s] * "
            "B_sparse[N_s,K_s]^T`；OP2 内部使用转置坐标 "
            "`(M,N,K)=(N_s,M_s,K_s)`。"
        ),
        "",
        "## 测试标准",
        "",
        "- pure matmul：`ElementC=void`，`beta=0`，不读取 C。",
        "- 每个 shape 使用持久化、确定性随机 A/B/SFA/SFB 文件；所有配置读取同一份输入。",
        "- baseline：5 个独立样本，每样本 1 次 warmup、10 次计时。",
        "- 广搜：每个 shape 1,080 个合法配置，共 14,040 条结果，失败 0。",
        "- Top20：每个候选 5 个独立样本；0.5% 窗口内至多 4 个候选追加 20 轮。",
        "- 最终 best：25 个独立样本，每样本 1 次 warmup、10 次计时。",
        "- 每个进程启动前等待 GPU 连续空闲，并轮换 shape/候选顺序。",
        "",
        "## 结果",
        "",
        (
            "|Shape|Dense M,N,K|Sparse M,N,K|FLOPs差|Bytes差|"
            "Baseline ms|Best ms|提升|Best配置|Stage|SMEM KiB|95% CI|"
        ),
        (
            "|---:|---|---|---:|---:|---:|---:|---:|---|"
            "---:|---:|---:|"
        ),
    ]
    for row in rows:
        config = (
            f"S{row['sm_mode']} "
            f"{row['tile_m']}x{row['tile_n']}x{row['tile_k']} "
            f"C{row['cluster']} E{row['epilogue']}"
        )
        lines.append(
            f"|{row['shape']}|"
            f"{row['dense_m']},{row['dense_n']},{row['dense_k']}|"
            f"{row['sparse_m']},{row['sparse_n']},{row['sparse_k']}|"
            f"{float(row['flop_difference_pct']):+.4f}%|"
            f"{float(row['byte_difference_pct']):+.4f}%|"
            f"{float(row['baseline_mean_ms']):.6f}|"
            f"{float(row['best_mean_ms']):.6f}|"
            f"{float(row['improvement_pct']):.2f}%|{config}|"
            f"{row['stages']}|{row['kernel_smem_kib']}|"
            f"±{float(row['best_ci95_half_ms']):.6f}|"
        )

    lines.extend(
        [
            "",
            "## 输入完整性",
            "",
            "- 共 13 个 shape、52 个 operand 文件，合计约 7.5 GiB。",
            "- 文件 manifest 中的 52 个大小均与实际文件一致。",
            "- FP4 文件抽样具有 256 个 byte 值，熵约 8.00 bit/byte。",
            "- scale 文件抽样具有 71 个 byte 值，熵约 6.14 bit/byte。",
            "",
            "## GPU 状态",
            "",
            f"- 最终 325 个样本的启动前 P-state：`{', '.join(pstates)}`。",
            (
                f"- 启动前空闲 SM 时钟范围：`{min(clocks)}-"
                f"{max(clocks)} MHz`。该字段是启动前 idle 状态，"
                "不等同于 kernel 执行期间时钟。"
            ),
            f"- 温度范围：`{min(temperatures)}-{max(temperatures)} C`。",
            "",
            "## 固定产物",
            "",
            "- `op2_eq_corrected_baseline.cu`：修正 shape 使用的 pure-matmul baseline 源码。",
            "- `op2_eq_corrected_best.cu`：按 13 个 shape 分支到最终 winner 的源码。",
            "- `equivalent_shapes.csv`：修正后的逻辑 shape、内部坐标和等价误差。",
            "- `random_datasets/manifest.csv`：固定随机输入清单。",
            "- `search_space/results/`：14,040 行广搜结果与 Top20。",
            "- `results/standard_search_20260727/`：baseline、Top20、tiebreak 和最终 25 轮原始数据。",
            "- `results/op2_eq_corrected_baseline_vs_best.csv`：统一对比表。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"generated {COMPARISON}")
    print(f"generated {REPORT}")


if __name__ == "__main__":
    main()
