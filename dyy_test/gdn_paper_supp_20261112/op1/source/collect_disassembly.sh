#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cuda=/usr/local/cuda-13.0/bin/cuobjdump
mkdir -p build/disassembly
for path in bin/*; do
  name="$(basename "$path")"
  "$cuda" --dump-resource-usage "$path" >"build/disassembly/$name.resources.txt"
  "$cuda" --dump-sass "$path" | gzip -c >"build/disassembly/$name.sass.txt.gz"
done
