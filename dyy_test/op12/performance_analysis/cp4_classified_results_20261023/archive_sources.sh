#!/usr/bin/env bash
set -euo pipefail
task_base=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis
archive_path="$task_base/_superseded/cp4_frequency_20261023"
test "$(realpath "$task_base")" = "$task_base"
names=(cp4_inchain_frequency_retest_20261023 cp4_cold_l2_lock1845_20261023 cp4_inchain_frequency_retest_20261023_transport.tar.gz cp4_cold_l2_lock1845_20261023_transport.tar.gz)
for name in "${names[@]}"; do
    test "$(realpath "$task_base/$name")" = "$task_base/$name"
    test ! -e "$archive_path/$name"
done
mkdir -p "$archive_path"
test "$(realpath "$archive_path")" = "$task_base/_superseded/cp4_frequency_20261023"
for name in "${names[@]}"; do
    mv -- "$task_base/$name" "$archive_path/$name"
    printf 'Archived: %s\n' "$name"
done
