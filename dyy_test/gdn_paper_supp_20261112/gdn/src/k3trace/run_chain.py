"""Measure pinned GDN CP with an output-preserving K2/K3 factorial."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / "config.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--s", type=int, choices=[8192, 32768], required=True)
parser.add_argument("--partitions", type=int, choices=[4, 8], required=True)
parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--arms", default="B0,K2,K3,K23", choices=["B0,K3", "B0,K2,K3,K23"])
a = parser.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
assert a.s % (a.partitions * 64) == 0
chunk_len = a.s // a.partitions
arms = a.arms.split(",")

def cmd(*args):
    return subprocess.check_output(args, text=True).strip()

def sha(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(fh)

def digest_stream(fh):
    digest = hashlib.sha256()
    for buf in iter(lambda: fh.read(8 << 20), b""):
        digest.update(buf)
    return digest.hexdigest()

def save(name, record):
    path = a.out / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)

repo = Path(cfg["host"]["flashinfer_repo"])
assert cmd("git", "-C", str(repo), "rev-parse", "HEAD") == cfg["pin_commit"]
assert not cmd("git", "-C", str(repo), "status", "--porcelain")
assert sha(repo / cfg["source_relpath"]) == cfg["source_sha256"]
sys.path[:0] = [str(repo), str(root / "generated")]
os.environ["FLASHINFER_WORKSPACE_BASE"] = str(a.out.resolve() / "jit_cache")
import torch
import cutlass.cute as cute
import flashinfer
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
from variants_k2opt import K2OptScaleNEarlyFusedM
from variants_k3opt import K3OptCoalescedStore

assert Path(flashinfer.__file__).resolve().is_relative_to(repo)
torch.set_num_threads(1)
prop = torch.cuda.get_device_properties(0)
assert prop.name == "NVIDIA B200" and prop.major == 10 and prop.multi_processor_count == 148
inp = Path(cfg["host"]["frozen_inputs"]) / f"S{a.s}" / "inputs.pt"
file_hash = sha(inp)
assert file_hash == json.loads((inp.parent / "storage_hash.json").read_text())["inputs.pt"]
manifest = json.loads((inp.parent / "inputs_manifest.json").read_text())
data = torch.load(inp, map_location="cpu", weights_only=True)
for buf, expected in [*zip(data["buffers"], manifest["buffers"]), (data["shared"], manifest["shared"])]:
    for key, tensor in buf.items():
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == expected[key]["sha256"], key
buffers = [{**{key: value.cuda() for key, value in buf.items()},
            "o": torch.empty((a.s, 16, 128), dtype=torch.float16, device="cuda"),
            "state": torch.empty((1, 16, 128, 128), dtype=torch.float32, device="cuda")}
           for buf in data["buffers"]]
shared = {key: value.cuda() for key, value in data["shared"].items()}
assert len(buffers) == 4
del data
initial = shared["initial"].clone()
evict = torch.empty(4 * int(prop.L2_cache_size), dtype=torch.uint8, device="cuda")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
torch.cuda.synchronize()
base_k2 = cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100
base_k3 = cp.CPDeltaRuleFixupUtcmmaSm100
orig_get, orig_compile = cp.get_cached_compile, cp.cached_compile
compiled, binaries = {}, {}
last_fixed = None

def get(kernel, options):
    if isinstance(kernel, (base_k2, base_k3)):
        return compiled.get(type(kernel))
    return orig_get(kernel, options)

def compile_kernel(kernel, *args, compile_options):
    if not isinstance(kernel, (base_k2, base_k3)):
        return orig_compile(kernel, *args, compile_options=compile_options)
    cls = type(kernel)
    dest = a.out / "binary" / cls.__name__
    dest.mkdir(parents=True)
    options = (*compile_options, cute.KeepPTX(True), cute.KeepCUBIN(True), cute.KeepSASS(True), DumpDir(str(dest.resolve())))
    fn = cute.compile[options](kernel, *args)
    def wrapped(*runtime):
        global last_fixed
        if isinstance(kernel, base_k3):
            last_fixed = runtime[4]
        return fn(*runtime)
    compiled[cls] = wrapped
    cubin = next(dest.glob("*.cubin"))
    sass = next(dest.glob("*.sass")).read_text()
    binaries[cls.__name__] = {
        "sha256": sha(cubin), "resource_usage": cmd(cfg["host"]["cuobjdump"], "--dump-resource-usage", str(cubin)),
        "local_ops_static": len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass)),
        "stg_static": len(re.findall(r"\bSTG\.", sass)),
    }
    save("COMPILE_AUDIT.json", binaries)
    return wrapped

cp.get_cached_compile, cp.cached_compile = get, compile_kernel

def invoke(index):
    b = buffers[index]
    return cp.cp_delta_rule_dsl_sm100(
        b["o"], b["state"], b["q"], b["k"], b["v"], shared["gate"], shared["beta"],
        shared["cu_seqlens"], 1 / math.sqrt(128), initial_state=shared["initial"],
        max_seqlen=a.s, cp_chunk_len=chunk_len)

def snapshot(index):
    b = buffers[index]
    return (b["o"], b["state"], last_fixed)

def kernel_nodes(path, expected_symbols):
    raw = path.read_text()
    nodes = list(re.finditer(r'"(graph_\d+_node_\d+)"\[(.*?)\];', raw, re.S))
    assert len(nodes) == 7, (path, len(nodes))
    assert "FillFunctor" in nodes[0].group(2), path
    assert all("EVENT_RECORD" in nodes[i].group(2) for i in (1, 6)), path
    edges = re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', raw)
    assert set(edges) == set(zip([n.group(1) for n in nodes[:-1]], [n.group(1) for n in nodes[1:]])), path
    kernels = nodes[2:6]
    result = []
    for node, symbol in zip(kernels, expected_symbols):
        body = node.group(2)
        assert 'label="{KERNEL' in body and symbol in body, (symbol, body)
        config = body.splitlines()[1].rsplit('\\<\\<\\<', 1)[1].split('\\>')[0].split(',')
        result.append({"symbol": symbol, "grid_x": int(config[0]), "threads": int(config[1]), "dynamic_smem": int(config[2])})
    assert [x["grid_x"] for x in result] == [a.s // 64 * 16, a.partitions * 16, 32, a.partitions * 16], result
    assert [x["threads"] for x in result] == [128, 384, 256, 384], result
    return result

save("IDENTITY.json", dict(S=a.s, P=a.partitions, C=chunk_len, arms=arms, trial=a.trial,
    repo_pin=cfg["pin_commit"], inputs_sha256=file_hash, input_signature=manifest["content_signature_sha256"],
    runner_sha256=sha(__file__), input_manifest=manifest, gpu=prop.name, sms=prop.multi_processor_count,
    evict_bytes=evict.numel(), python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
    dsl=importlib.metadata.version("nvidia-cutlass-dsl"), timing="CUDA Event inside graph around four kernels",
    clock_policy="natural clocks", samples=a.samples, warmup=a.warmup))
graphs, gold, audit = {}, {}, {}
for arm in arms:
    k2 = K2OptScaleNEarlyFusedM if arm in {"K2", "K23"} else base_k2
    k3 = K3OptCoalescedStore if arm in {"K3", "K23"} else base_k3
    cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100 = k2
    cp.CPDeltaRuleFixupUtcmmaSm100 = k3
    cp._get_mn_precompute_kernel.cache_clear()
    cp._get_fixup_kernel.cache_clear()
    for index in range(4):
        with torch.cuda.stream(stream):
            invoke(index)
            if arm == "B0":
                gold[index] = tuple(value.clone() for value in snapshot(index))
            else:
                assert all(torch.equal(x, y) for x, y in zip(snapshot(index), gold[index])), (arm, index, "eager correctness")
            begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            begin.record()
            end.record()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            evict.fill_(90)
            begin.record()
            invoke(index)
            end.record()
        fixed = last_fixed
        path = a.out / f"{arm}_b{index}.dot"
        graph.debug_dump(str(path))
        graph.instantiate()
        audit[f"{arm}:{index}"] = kernel_nodes(path, ["CPDeltaRuleTPrecomputeSm100", k2.__name__, k3.__name__, "CPDeltaRulePrefillTcgen05Sm100"])
        graphs[arm, index] = (graph, begin, end, fixed)
save("GRAPH_AUDIT.json", audit)

def replay(arm, index, check=True):
    graph, begin, end, fixed = graphs[arm, index]
    graph.replay()
    end.synchronize()
    elapsed = begin.elapsed_time(end) * 1000
    if check:
        b = buffers[index]
        assert all(torch.equal(x, y) for x, y in zip((b["o"], b["state"], fixed), gold[index])), (arm, index, "replay correctness")
    return elapsed

for index in range(4):
    for arm in arms:
        replay(arm, index)
save("CORRECTNESS.json", {"output": "exact", "final_state": "exact", "k3_fixed_state": "exact", "buffers": 4})
for i in range(a.warmup):
    for arm in arms:
        replay(arm, i % 4, check=False)
if len(arms) == 4:
    orders = [(0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1)]
else:
    orders = [(0, 1), (1, 0)]
with (a.out / "EVENT_SAMPLES.csv").open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["trial", "block", "position", "arm", "buffer", "cp_us"])
    for block in range(a.samples):
        order = orders[(block + a.trial) % len(orders)]
        for position, arm_id in enumerate((*order, *reversed(order))):
            arm = arms[arm_id]
            writer.writerow([a.trial, block, position, arm, block % 4, replay(arm, block % 4)])
        fh.flush()
assert torch.equal(initial, shared["initial"])
save("DONE.json", {"samples": a.samples, "arms": arms, "event_rows": 2 * len(arms) * a.samples})
