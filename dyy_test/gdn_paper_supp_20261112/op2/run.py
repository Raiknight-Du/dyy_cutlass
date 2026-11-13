"""Run the paired OP2 experiment after acquiring the shared GPU lock."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parent
REPO = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
DATASETS = REPO / "dyy_test/op2/random_datasets"
BINARY = ROOT / "bin/op2_schedule"
NCU = Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
SMI_FIELDS = "timestamp,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used"

CALIBRATION = [
    (8, 4096, 8192, 4096, DATASETS),
    (11, 768, 16384, 256, DATASETS),
]
# Arguments use transposed (N_logical, M_logical, K) coordinates.
HOLDOUT = [(8 if n == 4096 else 11, n, m, k, None)
           for m, n in [(8192, 4096), (32768, 768)] for k in [512, 1536, 3072]]


def idle():
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    util = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return not apps and util.isdecimal() and int(util) <= 2


def wait_idle(seconds):
    stop = time.monotonic() + seconds
    consecutive = 0
    while time.monotonic() < stop:
        consecutive = consecutive + 1 if idle() else 0
        if consecutive == 2:
            return
        time.sleep(1)
    raise RuntimeError("No uncontended GPU window before timeout")


def logged(command, path, timeout, telemetry=True):
    (path.with_suffix(".command")).write_text(shlex.join(str(x) for x in command) + "\n")
    with path.open("x") as stream:
        if not telemetry:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  check=True, timeout=timeout)
        with path.with_suffix(".telemetry.csv").open("x") as gpu_log:
            gpu_command = ["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv", "-lms", "200"]
            monitor = subprocess.Popen(gpu_command, stdout=gpu_log, stderr=subprocess.STDOUT)
            try:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                stop = time.monotonic() + timeout
                foreign = set()
                with path.with_suffix(".compute_apps.log").open("x") as apps_log:
                    while child.poll() is None:
                        if time.monotonic() >= stop:
                            child.kill()
                            child.wait()
                            raise subprocess.TimeoutExpired(command, timeout)
                        apps = subprocess.check_output(
                            ["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], text=True, timeout=5
                        ).strip()
                        apps_log.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} "
                                       f"expected_pid={child.pid} {apps!r}\n")
                        for row in apps.splitlines():
                            pid = row.split(",", 1)[0].strip()
                            if pid.isdecimal() and int(pid) != child.pid and command[0] != str(NCU):
                                foreign.add(row)
                        time.sleep(0.12)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
                if foreign:
                    path.with_suffix(".CONTAMINATED").write_text("\n".join(sorted(foreign)) + "\n")
            finally:
                monitor.terminate()
                monitor.wait(timeout=5)


def gemm_args(case, session, groups, iterations, warmups):
    sid, m, n, k, dataset = case
    options = [f"--shape-id={sid}", f"--m={m}", f"--n={n}", f"--k={k}",
               f"--session={session}", f"--groups={groups}", f"--iterations={iterations}",
               f"--warmups={warmups}"]
    if dataset:
        options.append(f"--dataset-root={dataset}")
    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "calibration", "holdout", "profile", "trace", "dense"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--wait-idle", type=int, default=12)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--session-start", type=int, default=0)
    parser.add_argument("--only-shape", type=int, choices=[8, 11])
    parser.add_argument("--only-k", type=int)
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        parser.error("label must be alphanumeric or hyphen separated")
    cases = HOLDOUT if args.phase in {"holdout", "dense"} else CALIBRATION
    if args.phase == "dense":
        cases = [case for case in cases if case[0] == 8 and case[3] in {512, 3072}]
    if args.only_shape:
        cases = [case for case in cases if case[0] == args.only_shape]
    if args.only_k:
        cases = [case for case in cases if case[3] == args.only_k]
    if not cases:
        parser.error("No selected cases")
    if args.sessions < 1 or args.session_start < 0:
        parser.error("sessions must be positive and session-start nonnegative")
    if args.phase == "profile" and not NCU.is_file():
        parser.error(f"ncu unavailable: {NCU}")
    binary = {
        "trace": ROOT / "bin/op2_trace",
        "dense": ROOT / "bin/op2_dense_reference",
    }.get(args.phase, BINARY)
    if not binary.is_file():
        parser.error(f"Binary not ready: {binary}")
    target = ROOT / "runs" / args.label
    target.mkdir(parents=True, exist_ok=False)
    (target / "manifest.json").write_text(json.dumps({
        "phase": args.phase, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cases_kernel_coordinates": [list(map(str, case)) for case in cases],
        "treatment": (
            "mode0 dense over the same masked operands; mode1 native sparse; all scales 1"
            if args.phase == "dense" else
            "after issuing stage k, both producer CTAs add a 512-cycle inter-issue gap if another stage follows"
        ),
        "binary": str(binary), "natural_gpu_clocks": True,
    }, indent=2) + "\n")
    logged(["nvidia-smi", "-q"], target / "environment.log", 20, telemetry=False)
    logged(["sha256sum", str(binary)], target / "binary.log", 20, telemetry=False)

    if args.phase == "trace":
        for session in range(args.session_start, args.session_start + args.sessions):
            for case in cases:
                for mode in [session % 2, 1 - session % 2]:
                    log = target / f"s{case[0]}-p{session}-mode{mode}.log"
                    wait_idle(args.wait_idle)
                    logged([str(binary), *gemm_args(case, session, 1, 1, 1),
                            f"--profile-mode={mode}"], log, 75)
                    if log.with_suffix(".CONTAMINATED").exists():
                        raise RuntimeError(f"Another compute process overlapped with {log}")
        return

    if args.phase == "profile":
        for case in cases:
            for mode in [0, 1]:
                sid = case[0]
                prefix = target / f"s{sid}-mode{mode}"
                command = [str(NCU), "--target-processes", "all", "--kernel-name-base", "demangled",
                           "--kernel-name", "regex:cutlass::device_kernel<cutlass::gemm", "--launch-count", "1",
                           "--section", "LaunchStats", "--section", "Occupancy",
                           "--metrics", "gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,"
                           "sm__sass_inst_executed_op_tma_ld.sum,sm__sass_inst_executed_op_tma_st.sum,"
                           "sm__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,dram__bytes_read.sum",
                           "--export", str(prefix), str(BINARY),
                           *gemm_args(case, 0, 1, 1, 1), f"--profile-mode={mode}"]
                wait_idle(args.wait_idle)
                logged(command, prefix.with_suffix(".log"), 120)
                report = prefix.with_suffix(".ncu-rep")
                if not report.is_file():
                    raise RuntimeError(f"No NCU report for {prefix}")
                logged([str(NCU), "--import", str(report), "--page", "raw", "--csv"],
                       prefix.with_suffix(".csv"), 45, telemetry=False)
        return

    session_count = 1 if args.phase == "smoke" else args.sessions
    for session in range(args.session_start, args.session_start + session_count):
        for case in cases[session % len(cases):] + cases[:session % len(cases)]:
            sid, m, n, k, _ = case
            log = target / f"s{sid}-M{m}-N{n}-K{k}-p{session}.log"
            smoke = args.phase == "smoke"
            command = [str(binary), *gemm_args(case, session, 2 if smoke else 20,
                                              8 if smoke else 64, 2 if smoke else 20)]
            wait_idle(args.wait_idle)
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {shlex.join(command)}", flush=True)
            logged(command, log, 100)
            if log.with_suffix(".CONTAMINATED").exists():
                raise RuntimeError(f"Another compute process overlapped with {log}")
            if not any(line.startswith("VERIFY,") for line in log.read_text().splitlines()):
                raise RuntimeError(f"No output correctness record for {log}")


if __name__ == "__main__":
    main()
