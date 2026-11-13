#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/gdn_paper_supp_20261112/op1
case "$1" in
  smoke)
    set -- --out smoke_shape7 --sessions 1 --repeats 2 --shapes 7
    ;;
  packing_short)
    set -- --out packing_shape7_10 --sessions 3 --repeats 20 --shapes 7 10 --resume
    ;;
  packing_large)
    set -- --out packing_shape1 --sessions 3 --repeats 20 --shapes 1 --timing packing-large --resume
    ;;
  intervention_pilot)
    set -- --out intervention_smoke --sessions 1 --repeats 2 --shapes 7 10 --timing intervention --resume
    ;;
  intervention_calibration)
    set -- --out intervention_shape7_10 --sessions 3 --repeats 20 --shapes 7 10 --timing intervention --resume
    ;;
  intervention_holdout)
    set -- --out intervention_holdout --sessions 3 --repeats 20 --shapes 21 22 23 24 25 26 --timing intervention --resume
    ;;
  *)
    echo "Unknown matrix name" >&2
    exit 2
    ;;
esac
flock -n ../gpu.lock python3 -u source/run_matrix.py "$@"
