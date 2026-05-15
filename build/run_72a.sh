#!/bin/bash

# --- 配置参数 ---
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

VMLINUX_EXE="./examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"

# 定义日志文件
LOG_FILE="ncu_72a_log.txt"

# 检查可执行文件
if [ ! -f "$VMLINUX_EXE" ]; then
    echo "错误: 找不到可执行文件 $VMLINUX_EXE"
    exit 1
fi

# 清空或创建日志文件，并写入开始时间
echo "=== NCU Batch Job Started ===" > "$LOG_FILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "Location: Guangdong, Guangzhou" >> "$LOG_FILE"
echo "=============================" >> "$LOG_FILE"
echo "开始执行 NCU 性能分析任务..."

# --- 执行逻辑 ---
for item in "${TEST_LIST[@]}"; do
    IFS=' ' read -r shape_num M N K <<< "$item"
    OUTPUT_FILE="ncu_72a_shape${shape_num}"
    
    # === 新增逻辑：检查文件是否已存在 ===
    if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
        echo "跳过 Shape ${shape_num}: 文件 ${OUTPUT_FILE}.ncu-rep 已存在"
        # 同时记录到日志文件中，保持日志完整性
        echo "" >> "$LOG_FILE"
        echo ">>> Shape ${shape_num} SKIPPED <<<" >> "$LOG_FILE"
        echo "Reason: File ${OUTPUT_FILE}.ncu-rep already exists." >> "$LOG_FILE"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
        continue
    fi
    # =====================================
    
    echo "----------------------------------------"
    echo "Processing Shape ${shape_num}: M=${M}, N=${N}, K=${K}"
    
    # 写入日志：当前任务开始的时间位置
    echo "" >> "$LOG_FILE"
    echo ">>> Shape ${shape_num} Start <<<" >> "$LOG_FILE"
    echo "Time Location: $(date '+%Y-%m-%d %A %H:%M:%S')" >> "$LOG_FILE"
    echo "Parameters: --m=${M} --n=${N} --k=${K}" >> "$LOG_FILE"
    echo "Command: ncu -o ${OUTPUT_FILE} -f --section SpeedOfLight ${VMLINUX_EXE} ..." >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    
    # 执行 NCU 命令
    if ncu -o "$OUTPUT_FILE" -f --set full -c 2 "$VMLINUX_EXE" --m="$M" --n="$N" --k="$K" >> "$LOG_FILE" 2>&1; then
        echo "Success: Shape ${shape_num} completed."
        echo "Status: SUCCESS" >> "$LOG_FILE"
    else
        echo "Error: Shape ${shape_num} failed."
        echo "Status: FAILED" >> "$LOG_FILE"
    fi
    
    # 写入任务结束时间
    echo "Time Finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo ">>> Shape ${shape_num} End <<<" >> "$LOG_FILE"
done

echo "----------------------------------------"
echo "所有任务执行完毕。完整日志已保存至 ${LOG_FILE}"

# 写入结束时间
echo "" >> "$LOG_FILE"
echo "=== NCU Batch Job Finished ===" >> "$LOG_FILE"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"