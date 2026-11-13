#!/bin/bash
set -u
cd /home/ubuntu/gdn_paper_supp_20261112
while ps -p 2388309 -o stat= | grep -qv '^Z'; do
  if grep -q 'END-OF-QUEUE' queue_1113.log; then break; fi
  sleep 2
done
run() { local title=$1; shift; echo "$(date -Iseconds) BEGIN $title"; "$@"; local status=$?; echo "$(date -Iseconds) END $title $status"; }
run D1-holdout flock -w 20 gpu.lock python3 op1/source/run_matrix.py --out op1/intervention_holdout --sessions 3 --repeats 20 --shapes 21 22 23 24 25 26 --timing intervention
run D1-NCU flock -w 20 gpu.lock bash op1/source/profile_shape7.sh ../profile_shape7
for t in 41 42 43; do
  run G2-P4-"$t" bash gdn/run_remote.sh formal 32768 4 "$t" B0,K2,K3,K23
done
for s in 8192 32768; do
  for t in 41 42 43; do
    run G2-P8-"$s"-"$t" bash gdn/run_remote.sh formal "$s" 8 "$t" B0,K2,K3,K23
  done
done
echo "$(date -Iseconds) END-OF-QUEUE"
