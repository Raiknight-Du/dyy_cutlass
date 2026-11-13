from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import subprocess
import time
from pathlib import Path


def gpu_apps():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    return {
        int(line.split(",")[0]): line.strip()
        for line in result.splitlines()
        if line.strip() and line.split(",")[0].strip().isdigit()
    }


def run_checked(executable, command, out, timeout):
    apps = gpu_apps()
    if apps:
        raise RuntimeError(f"GPU occupied before launch: {apps}")
    with out.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen([str(executable), *command], stdout=stream,
                                stderr=subprocess.STDOUT)
        start = time.monotonic()
        while proc.poll() is None:
            time.sleep(0.6)
            apps = {pid: line for pid, line in gpu_apps().items() if pid != proc.pid}
            if apps:
                proc.terminate()
                proc.wait(timeout=10)
                raise RuntimeError(f"Foreign GPU workload during {out.name}: {apps}")
            if time.monotonic() - start > timeout:
                proc.terminate()
                proc.wait(timeout=10)
                raise TimeoutError(out)
    if proc.returncode != 0:
        raise RuntimeError(f"{out}: exit {proc.returncode}")


def parse(path):
    rows, digests, meta, packed_bytes = [], {}, [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        kind, _, values = line.partition(",")
        values = values.split(",")
        if kind == "TIME":
            r, group, mode, launches, milliseconds = values
            rows.append((int(r), int(group), mode, int(launches), float(milliseconds)))
        if kind == "DIGEST":
            index, left, right = values
            digests[int(index)] = [left, right]
        if kind == "META":
            meta = values
        if kind == "PACKED_BYTES":
            packed_bytes = [int(value) for value in values]
    return rows, digests, meta, packed_bytes


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--shapes", type=int, nargs="+", default=[1, 7, 10])
    parser.add_argument("--timing", choices=["packing", "packing-large", "intervention"], default="packing")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    raw = args.out / "raw"
    raw.mkdir(exist_ok=args.resume)
    seed_index = 2011112
    output_rows, block_rows, digest_rows = [], [], []
    base = Path(__file__).resolve().parent
    binaries = base / "bin"
    gpu_log = args.out / "gpu_500ms.csv"
    metadata = args.out / "metadata.json"
    config = {
        "timing": args.timing, "sessions": args.sessions, "repeats": args.repeats,
        "shapes": args.shapes, "fixed_input_seed_index": seed_index,
        "hostname": os.uname().nodename,
    }
    if metadata.exists() and json.loads(metadata.read_text()) != config:
        raise RuntimeError("Resume requires exactly the original experiment settings")
    metadata.write_text(json.dumps(config, indent=2))
    with gpu_log.open("a", encoding="utf-8") as stream:
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,name,clocks.sm,clocks.mem,"
             "temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv", "--loop-ms=500"], stdout=stream, stderr=subprocess.STDOUT)
        try:
            for session in range(args.sessions):
                for shape in args.shapes[session % len(args.shapes):] + args.shapes[:session % len(args.shapes)]:
                    if args.timing.startswith("packing"):
                        order = ["baseline", "packed", "packed", "baseline"]
                    else:
                        order = ["baseline", "packed", "packed_late", "baseline_late",
                                 "baseline_late", "packed_late", "packed", "baseline"]
                    if session % 2:
                        first = list(reversed(order[:len(order) // 2]))
                        order = first + list(reversed(first))
                    prefix = {"packing":"reuse", "packing-large":"reuse_large",
                              "intervention":"intervention"}[args.timing]
                    samples, digest_reference = {}, None
                    for position, arm in enumerate(order):
                        path = raw / f"session{session}_shape{shape}_{position}_{arm}.log"
                        command = [f"--shape-id={shape}", f"--repeats={args.repeats}",
                                   f"--seed-index={seed_index}"]
                        completed = path.with_suffix(".ok")
                        resumed = args.resume and completed.exists()
                        if not resumed:
                            if path.exists():
                                path.rename(path.with_suffix(f".interrupted_{time.time_ns()}.log"))
                            timeout = 30 if args.timing == "intervention" else 120
                            run_checked(binaries / f"{prefix}_{arm}", command, path, timeout)
                        rows, digests, meta, packed_bytes = parse(path)
                        if len(digests) != 16 or len(set(map(tuple, digests.values()))) != 16:
                            raise RuntimeError(f"{path}: insufficient distinct output digests")
                        factor = 1 if args.timing == "intervention" else (3 if arm == "baseline" else 9)
                        if not meta or int(meta[0]) != shape or len(rows) != factor * args.repeats:
                            raise RuntimeError(f"{path}: timing matrix is incomplete or wrong shape")
                        if digest_reference is None:
                            digest_reference = digests
                        if digests != digest_reference:
                            raise RuntimeError(f"{path}: bitwise digests differ")
                        if not resumed:
                            completed.touch()
                        for index, digest in digests.items():
                            digest_rows.append((session, shape, position, arm, index, *digest))
                        write_csv(args.out / "digests.csv",
                                  ["session", "shape", "position", "arm", "input_index", "sum", "xor"],
                                  digest_rows)
                        for reuse, group, mode, launches, elapsed in rows:
                            key = reuse, group, arm, mode
                            samples.setdefault(key, []).append(elapsed)
                            output_rows.append((session, shape, position, arm, reuse, group, mode,
                                                launches, elapsed, *packed_bytes) if packed_bytes else
                                               (session, shape, position, arm, reuse, group, mode,
                                                launches, elapsed, 0, 0))
                        write_csv(args.out / "timings.csv", [
                            "session", "shape", "position", "arm", "reuse", "group", "mode",
                            "graph_launches", "ms_per_chain", "packed_a_bytes", "packed_b_bytes"
                        ], output_rows)
                        print(f"{path.name}: {len(rows)} timing rows, matched {len(digests)} output digests",
                              flush=True)
                    for (reuse, group, arm, mode), values in samples.items():
                        block_rows.append((session, shape, reuse, group, arm, mode, len(values), stats.mean(values)))
                    write_csv(args.out / "process_pair_means.csv", [
                        "session", "shape", "reuse", "group", "arm", "mode", "processes", "ms_per_chain"
                    ], block_rows)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
