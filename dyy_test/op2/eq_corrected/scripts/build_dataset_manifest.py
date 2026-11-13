#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op2_eq_corrected")
SHAPES = ROOT / "equivalent_shapes.csv"
DATASETS = ROOT / "random_datasets"
MANIFEST = DATASETS / "manifest.csv"


def read_metadata(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as stream:
        for line in stream:
            key, value = line.rstrip().split("=", 1)
            result[key] = value
    return result


def main() -> None:
    with SHAPES.open(newline="", encoding="utf-8-sig") as stream:
        shapes = list(csv.DictReader(stream))

    rows = []
    for shape in shapes:
        op2_m = int(shape["op2_m"])
        op2_n = int(shape["op2_n"])
        op2_k = int(shape["op2_k"])
        directory = DATASETS / f"M{op2_m}_N{op2_n}_K{op2_k}"
        for operand in ("A", "B", "SFA", "SFB"):
            path = directory / f"{operand}.bin"
            metadata = read_metadata(Path(str(path) + ".meta"))
            actual_bytes = path.stat().st_size
            if actual_bytes != int(metadata["bytes"]):
                raise RuntimeError(f"Size mismatch: {path}")
            rows.append({
                "shape": shape["shape"],
                "sparse_m": shape["sparse_m"],
                "sparse_n": shape["sparse_n"],
                "sparse_k": shape["sparse_k"],
                "op2_m": op2_m,
                "op2_n": op2_n,
                "op2_k": op2_k,
                "operand": operand,
                "bytes": actual_bytes,
                "seed": metadata["seed"],
                "kind": metadata["kind"],
                "path": str(path),
            })

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {MANIFEST} with {len(rows)} operand records")


if __name__ == "__main__":
    main()
