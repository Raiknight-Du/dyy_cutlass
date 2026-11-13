#!/usr/bin/env python3
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OLD = ROOT / "dyy_test/m72a_m84a_voidc_baseline_best_20260722/matmul/M_72a_pure_matmul_variants"
NEW = ROOT / "dyy_test/op1/bin/op1_top20"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def kernels(binary):
    output = subprocess.check_output([CUOBJDUMP, "--dump-sass", str(binary)], text=True)
    result = {}
    current = None
    lines = []
    for line in output.splitlines():
        match = re.match(r"\s*Function : (.+)", line)
        if match:
            if current is not None:
                result[current] = "\n".join(lines)
            current = match.group(1)
            lines = []
        elif current is not None:
            if line.startswith("Fatbin elf code:") or line.startswith("Fatbin ptx code:"):
                result[current] = "\n".join(lines)
                current = None
                lines = []
            else:
                # Compare decoded instructions, not PC offsets or binary encodings, which
                # change when the same kernel is linked into a larger fatbin.
                if re.search(r"/\*[0-9a-fA-F]{4}\*/", line):
                    decoded = re.sub(r"/\*[^*]*\*/", "", line)
                    decoded = " ".join(decoded.split())
                    if decoded:
                        lines.append(decoded)
    if current is not None:
        result[current] = "\n".join(lines)
    return result


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def opcode_stream(text):
    opcodes = []
    for line in text.splitlines():
        tokens = line.replace(";", "").split()
        if not tokens:
            continue
        if tokens[0].startswith("@"):
            tokens = tokens[1:]
        if tokens:
            opcodes.append(tokens[0])
    return opcodes


def main():
    old = kernels(OLD)
    new = kernels(NEW)
    common = sorted(set(old) & set(new))
    mismatches = [name for name in common if old[name] != new[name]]
    opcode_matches = [name for name in common if opcode_stream(old[name]) == opcode_stream(new[name])]
    print(f"old_kernels={len(old)}")
    print(f"new_kernels={len(new)}")
    print(f"common_kernels={len(common)}")
    print(f"identical_common={len(common) - len(mismatches)}")
    print(f"identical_opcode_stream={len(opcode_matches)}")
    print(f"mismatches={len(mismatches)}")
    for name in mismatches[:2]:
        print(
            f"MISMATCH_SUMMARY,old_instructions={len(old[name].splitlines())},"
            f"new_instructions={len(new[name].splitlines())},"
            f"opcode_equal={opcode_stream(old[name]) == opcode_stream(new[name])}"
        )


if __name__ == "__main__":
    main()
