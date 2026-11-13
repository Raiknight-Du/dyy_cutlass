#!/usr/bin/env python3
import csv
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "op2_baseline.cu"
MEMBERS = HERE / "group_members.csv"
OUTPUT = HERE / "generated_groups"


def parse_variant(name: str) -> tuple[int, ...]:
    match = re.fullmatch(
        r"S([12])_M(\d+)_N(\d+)_K(\d+)_E(auto|(\d+)x(\d+))_C(\d+)x(\d+)x(\d+)",
        name,
    )
    if not match:
        raise ValueError(f"Unsupported variant name: {name}")
    sm_mode = int(match.group(1))
    tile_m = int(match.group(2))
    tile_n = int(match.group(3))
    tile_k = int(match.group(4))
    epi_auto = match.group(5) == "auto"
    epi_m = int(match.group(6)) if not epi_auto else 128
    epi_n = int(match.group(7)) if not epi_auto else 64
    cluster_m = int(match.group(8))
    cluster_n = int(match.group(9))
    cluster_k = int(match.group(10))
    return (
        sm_mode,
        tile_m,
        tile_n,
        tile_k,
        cluster_m,
        cluster_n,
        cluster_k,
        epi_auto,
        epi_m,
        epi_n,
    )


def config_alias(index: int, variant: str) -> str:
    values = parse_variant(variant)
    rendered = [
        str(value).lower() if isinstance(value, bool) else str(value)
        for value in values
    ]
    return f"using Config_{index:02d} = KernelConfig<{', '.join(rendered)}>;"


def main() -> None:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with MEMBERS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["group_name"]].append(
                (int(row["config_index"]), row["variant"])
            )

    baseline = BASELINE.read_text()
    alias_pattern = re.compile(
        r"using Config_00 = KernelConfig<[^;]+>;\n\n"
        r"using CanonicalConfig = Config_00;"
    )
    dispatch_pattern = re.compile(
        r'  if \(options\.config_index < 0 \|\| options\.config_index == 0\) '
        r'failures \+= run_config<Config_00>\("[^"]+", options\.shape_id, options\);'
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for group_name in sorted(groups):
        members = sorted(groups[group_name])
        indices = [index for index, _ in members]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"Duplicate indices in {group_name}: {indices}")

        aliases = "\n".join(
            config_alias(index, variant) for index, variant in members
        )
        canonical_index = members[0][0]
        alias_block = (
            f"{aliases}\n\n"
            f"using CanonicalConfig = Config_{canonical_index:02d};"
        )
        dispatch = "\n".join(
            "  if (options.config_index < 0 || "
            f"options.config_index == {index}) "
            f'failures += run_config<Config_{index:02d}>('
            f'"{variant}", options.shape_id, options);'
            for index, variant in members
        )

        source, alias_count = alias_pattern.subn(alias_block, baseline, count=1)
        source, dispatch_count = dispatch_pattern.subn(dispatch, source, count=1)
        if alias_count != 1 or dispatch_count != 1:
            raise RuntimeError(
                f"Template replacement failed for {group_name}: "
                f"aliases={alias_count}, dispatch={dispatch_count}"
            )

        path = OUTPUT / f"{group_name}.cu"
        path.write_text(source)
        for index, variant in members:
            manifest_rows.append(
                {
                    "group_name": group_name,
                    "config_index": index,
                    "variant": variant,
                    "source": path.name,
                }
            )
        print(f"generated {path.name}: {len(members)} configs")

    with (OUTPUT / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["group_name", "config_index", "variant", "source"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"generated {len(groups)} groups and {len(manifest_rows)} configs")


if __name__ == "__main__":
    main()
