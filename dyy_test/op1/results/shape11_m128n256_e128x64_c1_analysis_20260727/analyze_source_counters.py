#!/usr/bin/env python3
import csv
import pathlib


PATH = (
    pathlib.Path(__file__).resolve().parent / "ncu" / "e64_full_source.csv"
)
CUDA_SASS_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "ncu"
    / "e64_full_source_cuda_sass.csv"
)


def main():
    with PATH.open(newline="", encoding="utf-8-sig") as stream:
        next(stream)
        rows = list(csv.DictReader(stream))

    ranked = sorted(
        rows,
        key=lambda row: int(row.get("stall_long_sb", "0") or 0),
        reverse=True,
    )
    output = []
    for row in ranked:
        samples = int(row.get("stall_long_sb", "0") or 0)
        if not samples:
            continue
        output.append(
            {
                "address": row["Address"],
                "instruction": row["Source"].strip(),
                "long_sb_samples": samples,
                "all_samples": int(row.get("# Samples", "0") or 0),
            }
        )

    with (
        PATH.parent / "source_long_scoreboard_top.csv"
    ).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    for row in output[:20]:
        print(
            row["address"],
            row["long_sb_samples"],
            row["all_samples"],
            row["instruction"],
        )

    source_rows = []
    current_file = ""
    header = None
    with CUDA_SASS_PATH.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.reader(stream):
            if not raw:
                continue
            if raw[0] == "File Path":
                current_file = raw[1]
                header = None
            elif raw[0] == "Line No":
                header = raw
            elif header and raw[0].isdigit():
                values = dict(zip(header, raw))
                samples = int(values.get("stall_long_sb", "0") or 0)
                if samples:
                    source_rows.append(
                        {
                            "file": current_file,
                            "line": int(raw[0]),
                            "source": raw[1].strip(),
                            "long_sb_samples": samples,
                            "all_samples": int(
                                values.get("# Samples", "0") or 0
                            ),
                        }
                    )

    source_rows.sort(key=lambda row: row["long_sb_samples"], reverse=True)
    with (
        PATH.parent / "source_line_long_scoreboard.csv"
    ).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    print("CUDA source lines:")
    for row in source_rows[:20]:
        print(
            pathlib.Path(row["file"]).name,
            row["line"],
            row["long_sb_samples"],
            row["all_samples"],
            row["source"],
        )


if __name__ == "__main__":
    main()
