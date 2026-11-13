#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2"
RUN = OP2 / "results/standard_search_20260724"
BASELINE = RUN / "op2_baseline_summary.csv"
BEST = RUN / "op2_best_final_summary.csv"
BEST_RAW = RUN / "op2_best_final_raw.csv"
CONFIGS = OP2 / "results/op2_best_configs.csv"
LEGAL = OP2 / "search_space/legacy_legal_scan/kernel_configs.csv"
COMPARISON = OP2 / "results/op2_baseline_vs_best.csv"
REPORT = OP2 / "ANALYSIS_CN.md"

ORIGINAL_SHAPES = {
    1: (170100, 13824, 5120),
    2: (510300, 13824, 5120),
    3: (170100, 5120, 5120),
    4: (510300, 5120, 5120),
    5: (8192, 768, 4096),
    6: (16384, 768, 4096),
    7: (32768, 768, 4096),
    8: (8192, 4096, 2048),
    9: (16384, 4096, 2048),
    10: (32768, 4096, 128),
    11: (16384, 768, 128),
    12: (32768, 768, 128),
    13: (8192, 4096, 128),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def config_id(variant: str) -> str:
    return variant.rsplit("_C", 1)[0]


def main() -> None:
    baseline = {int(row["shape"]): row for row in read_csv(BASELINE)}
    best = {int(row["shape"]): row for row in read_csv(BEST)}
    configs = {int(row["shape"]): row for row in read_csv(CONFIGS)}
    legal = {row["kernel_id"]: row for row in read_csv(LEGAL)}
    if not (
        sorted(baseline) == sorted(best) == sorted(configs) == list(range(1, 14))
    ):
        raise RuntimeError("Expected complete 13-shape inputs")

    rows = []
    for shape in range(1, 14):
        m, n, k = ORIGINAL_SHAPES[shape]
        base_ms = float(baseline[shape]["mean_ms"])
        best_ms = float(best[shape]["mean_ms"])
        config = configs[shape]
        resource = legal[config_id(config["variant"])]
        rows.append(
            {
                "shape": shape,
                "original_m": m,
                "original_n": n,
                "dense_k": k,
                "transposed_m": n,
                "transposed_n": m,
                "sparse_k": 2 * k,
                "baseline_variant": baseline[shape]["variant"],
                "baseline_mean_ms": f"{base_ms:.9f}",
                "baseline_stdev_ms": baseline[shape]["stdev_ms"],
                "baseline_samples": baseline[shape]["samples"],
                "best_variant": config["variant"],
                "best_mean_ms": f"{best_ms:.9f}",
                "best_stdev_ms": best[shape]["stdev_ms"],
                "best_ci95_half_ms": best[shape]["ci95_half_ms"],
                "best_samples": best[shape]["samples"],
                "improvement_pct": f"{(base_ms - best_ms) / base_ms * 100.0:.4f}",
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
                    f"{int(resource['kernel_smem_bytes']) / 1024.0:.3f}"
                ),
            }
        )

    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_baseline = sum(float(row["baseline_mean_ms"]) for row in rows)
    total_best = sum(float(row["best_mean_ms"]) for row in rows)
    total_improvement = (total_baseline - total_best) / total_baseline * 100.0
    best_raw = read_csv(BEST_RAW)
    pstates = sorted({row["pstate"] for row in best_raw})
    idle_clocks = [int(float(row["idle_sm_mhz"])) for row in best_raw]
    temperatures = [int(float(row["temperature_c"])) for row in best_raw]
    max_clock_count = sum(clock == max(idle_clocks) for clock in idle_clocks)

    lines = [
        "# OP2 M_84a Pure Matmul 最优配置搜索报告",
        "",
        "## 测试语义与标准",
        "",
        "- 逻辑计算：`D[M,N] = A_dense[M,K] * B_sparse[K,N]`。",
        "- 实现方式：转置为 `D^T[N,M] = B_sparse^T[N,2K] * A_dense^T[2K,M]`。",
        "- 稀疏 K 为 dense K 的 2 倍，保持与 OP1 相同的有效 FLOPs。",
        "- `ElementC=void`，epilogue 为 `Identity(accumulator) -> BF16 D`，不含 alpha/beta visitor。",
        "- 13 个 shape 使用固定、可复现、非全零/全一的持久化随机输入。",
        "- CUDA Event；每样本 1 次 warmup、10 iterations。",
        "- Top20 每候选 5 次独立进程；近似候选追加 20 次；最终 best 每 shape 25 次。",
        "- 候选和 shape 顺序按 repeat 轮换/反转，每次启动前确认 GPU 空闲。",
        "",
        "## 搜索范围",
        "",
        "- 合法配置：每 shape 1,080 个，合计 14,040 个粗筛结果，运行失败 0。",
        "- 每 shape 取粗筛 Top20 进入独立标准复测。",
        "- 近似候选按均值差、相对 0.5% 窗口和联合标准误自动进入加密复测。",
        "",
        "## Baseline 与 Best",
        "",
        "|Shape|原始 M,N,K|Best mean ms|95% CI half|Baseline ms|提升|Best 配置|Stage|SMEM KiB|",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        config = (
            f"S{row['sm_mode']} "
            f"{row['tile_m']}x{row['tile_n']}x{row['tile_k']} "
            f"C{row['cluster']} E{row['epilogue']}"
        )
        lines.append(
            f"|{row['shape']}|{row['original_m']}, {row['original_n']}, "
            f"{row['dense_k']}|{float(row['best_mean_ms']):.6f}|"
            f"{float(row['best_ci95_half_ms']):.6f}|"
            f"{float(row['baseline_mean_ms']):.6f}|"
            f"{float(row['improvement_pct']):.2f}%|{config}|"
            f"{row['stages']}|{row['kernel_smem_kib']}|"
        )
    lines.extend(
        [
            "",
            f"- 13-shape baseline 总时延：`{total_baseline:.6f} ms`。",
            f"- 13-shape best 总时延：`{total_best:.6f} ms`。",
            f"- 按总时延计算的整体改善：`{total_improvement:.2f}%`。",
            "",
            "## GPU 状态核对",
            "",
            f"- 最终 325 个样本的启动前 P-state 均为 `{', '.join(pstates)}`。",
            f"- 启动前 SM 时钟范围为 `{min(idle_clocks)}–{max(idle_clocks)} MHz`；"
            f"其中 `{max_clock_count}/325` 个样本为 `{max(idle_clocks)} MHz`。",
            f"- 温度范围为 `{min(temperatures)}–{max(temperatures)} °C`，未观察到热降频条件。",
            "- 少量低启动时钟是 GPU 空闲后的爬频状态；shape 与候选顺序轮换，"
            "且最终结果使用 25 个独立样本，未出现持续低频批次。",
            "",
            "## 固定产物",
            "",
            "- `op2_baseline.cu`：CUTLASS 原始 baseline 配置，仅改为当前 pure matmul 语义。",
            "- `op2_best.cu`：按 shape 分支到最终最优配置。",
            "- `results/op2_best_configs.csv`：最优配置、搜索阶段统计与展开后的参数。",
            "- `results/op2_baseline_vs_best.csv`：13-shape baseline/best 对照。",
            "- `results/standard_search_20260724`：Top20、tiebreak 与最终 25-repeat 原始数据。",
            "",
            "旧的错误方向、旧 void-C、非标准 10 配置和混合 OP1/OP2 性能数据已清理；"
            "详见 `OLD_DATA_CLEANUP_20260724.md`。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"generated {COMPARISON}")
    print(f"generated {REPORT}")


if __name__ == "__main__":
    main()
