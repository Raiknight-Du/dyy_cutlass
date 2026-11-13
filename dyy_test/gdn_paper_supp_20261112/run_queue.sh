#!/bin/bash
set -u
cd /home/ubuntu/gdn_paper_supp_20261112
run() { local title=$1; shift; echo "$(date -Iseconds) BEGIN $title"; "$@"; local status=$?; echo "$(date -Iseconds) END $title $status"; }
for i in 21 22 23; do
  for s in 8192 32768; do
    run "G1-$s-$i" bash gdn/run_remote.sh formal "$s" 8 "$i" B0,K3
  done
done
run D1-smoke flock -w 20 gpu.lock python3 op1/source/run_matrix.py --out op1/intervention_smoke --sessions 1 --repeats 2 --shapes 7 10 --timing intervention
run D2-shape1 flock -w 20 gpu.lock python3 op1/source/run_matrix.py --out op1/packing_shape1 --sessions 3 --repeats 20 --shapes 1 --timing packing-large
run S1-calibration flock -w 20 gpu.lock bash -c 'cd op2 && python3 run.py calibration --label calibration-full-1113 --sessions 3'
run S2-dense flock -w 20 gpu.lock bash -c 'cd op2 && python3 run.py dense --label dense-1113 --sessions 3'
run D1-formal flock -w 20 gpu.lock python3 op1/source/run_matrix.py --out op1/intervention_formal --sessions 3 --repeats 20 --shapes 7 10 --timing intervention
run S1-holdout flock -w 20 gpu.lock bash -c 'cd op2 && python3 run.py holdout --label holdout-1113 --sessions 3'
run G2-pilot bash gdn/run_remote.sh pilot 32768 4 31 B0,K2,K3,K23
run G1-nsys-short bash gdn/run_remote.sh nsys 8192 8 31 B0,K3
run G1-nsys-long bash gdn/run_remote.sh nsys 32768 8 31 B0,K3
run G1-ncu-long bash gdn/run_remote.sh ncu 32768 8 31 B0,K3
run S1-trace flock -w 20 gpu.lock bash -c 'cd op2 && python3 run.py trace --label trace-1113'
run S1-NCU flock -w 20 gpu.lock bash -c 'cd op2 && python3 run.py profile --label profile-1113'
echo "$(date -Iseconds) END-OF-QUEUE"
