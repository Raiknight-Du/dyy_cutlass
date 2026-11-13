#!/usr/bin/env python3
import csv
import re
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2"
BASELINE = OP2 / "op2_baseline.cu"
WINNERS = OP2 / "results/standard_search_20260724/op2_final_winners.csv"
OUTPUT = OP2 / "op2_best.cu"
CONFIG_OUTPUT = OP2 / "results/op2_best_configs.csv"


def parse_variant(name: str) -> tuple[int, ...]:
    match = re.fullmatch(
        r"S([12])_M(\d+)_N(\d+)_K(\d+)_E(auto|(\d+)x(\d+))_C(\d+)x(\d+)x(\d+)",
        name,
    )
    if not match:
        raise ValueError(f"Unsupported variant name: {name}")
    epi_auto = match.group(5) == "auto"
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(8)),
        int(match.group(9)),
        int(match.group(10)),
        epi_auto,
        int(match.group(6)) if not epi_auto else 128,
        int(match.group(7)) if not epi_auto else 64,
    )


def alias(shape: int, variant: str) -> str:
    values = parse_variant(variant)
    rendered = [
        str(value).lower() if isinstance(value, bool) else str(value)
        for value in values
    ]
    return (
        f"using Config_S{shape:02d} = "
        f"KernelConfig<{', '.join(rendered)}>;"
    )


def main() -> None:
    with WINNERS.open(newline="") as stream:
        rows = sorted(
            csv.DictReader(stream), key=lambda row: int(row["shape"])
        )
    if [int(row["shape"]) for row in rows] != list(range(1, 14)):
        raise RuntimeError("Winner file must contain exactly shapes 1 through 13")

    source = BASELINE.read_text()
    alias_pattern = re.compile(
        r"using Config_00 = KernelConfig<[^;]+>;\n\n"
        r"using CanonicalConfig = Config_00;"
    )
    aliases = "\n".join(
        alias(int(row["shape"]), row["variant"]) for row in rows
    )
    alias_block = f"{aliases}\n\nusing CanonicalConfig = Config_S01;"
    source, alias_count = alias_pattern.subn(alias_block, source, count=1)

    dispatch_pattern = re.compile(
        r'  if \(options\.config_index < 0 \|\| options\.config_index == 0\) '
        r'failures \+= run_config<Config_00>\("[^"]+", options\.shape_id, options\);'
    )
    cases = ["  switch (options.shape_id) {"]
    for row in rows:
        shape = int(row["shape"])
        cases.append(
            f"    case {shape}: failures += run_config<Config_S{shape:02d}>"
            f'("{row["variant"]}", options.shape_id, options); break;'
        )
    cases.extend(
        [
            "    default:",
            '      std::cerr << "shape-id must be in [1,13]\\n";',
            "      return 1;",
            "  }",
        ]
    )
    dispatch_block = "\n".join(cases)
    source, dispatch_count = dispatch_pattern.subn(
        lambda _: dispatch_block, source, count=1
    )
    if alias_count != 1 or dispatch_count != 1:
        raise RuntimeError(
            f"Template replacement failed: aliases={alias_count}, "
            f"dispatch={dispatch_count}"
        )

    OUTPUT.write_text(source)
    CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_OUTPUT.open("w", newline="") as stream:
        fieldnames = list(rows[0]) + [
            "sm_mode",
            "tile_m",
            "tile_n",
            "tile_k",
            "cluster_m",
            "cluster_n",
            "cluster_k",
            "epilogue_auto",
            "epilogue_m",
            "epilogue_n",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = parse_variant(row["variant"])
            enriched = dict(row)
            for key, value in zip(fieldnames[-10:], values):
                enriched[key] = value
            writer.writerow(enriched)
    print(f"generated {OUTPUT}")
    print(f"generated {CONFIG_OUTPUT}")


if __name__ == "__main__":
    main()
