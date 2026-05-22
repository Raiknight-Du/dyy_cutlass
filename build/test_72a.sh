#!/bin/bash

TEST_LIST=(
    "13 8192 4096 128"
    "12 32768 768 128"
    "11 16384 768 128"
    "10 32768 4096 128"
    "9 16384 4096 2048"
    "8 8192 4096 2048"
    "7 32768 768 4096"
    "6 16384 768 4096"
    "5 8192 768 4096"
    "4 510300 5120 5120"
    "3 170100 5120 5120"
    "2 510300 13824 5120"
    "1 170100 13824 5120"
)

OUTPUT_FILE="result_72a.txt"
VMLINUX_EXE="./examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"
# Clear output file
> "$OUTPUT_FILE"

# Run tests
for item in "${TEST_LIST[@]}"; do
    IFS=' ' read -r shape_num M N K <<< "$item"
    echo "Running test: $item" | tee -a "$OUTPUT_FILE"
    $VMLINUX_EXE --m="$M" --n="$N" --k="$K" --iterations=100000 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "Tests completed. Results written to $OUTPUT_FILE"
