#!/bin/sh
# Same dataset and seed, --threads 1 vs N: compare metrics and wall time.
# Usage: sh examples/compare_threads.sh [threads...]  (default: 1 2 4 8)
# For hybrid-CPU pinning (e.g. P-cores vs E-cores), wrap with taskset, e.g.:
#   taskset -c 0,2,4,6,8,10,12,14 ./src/sjev train ... --threads 8
set -eu
cd "$(dirname "$0")/.."
[ -d examples/work ] || python3 tools/make_synthetic.py \
  --output-dir examples/work --train 2000 --validation 400 --test 400 --seed 7
# shellcheck disable=SC2086
for t in ${*:-1 2 4 8}; do
  echo "=== threads $t ==="
  ./src/sjev train examples/work/train.jsonl \
    --validation examples/work/validation.jsonl \
    --output "examples/work/compare_t$t.sjev" \
    --width 64 --rank 64 --context-tokens 192 --option-tokens 32 \
    --epochs 8 --batch-size 64 --seed 7 --threads "$t" | tail -n 2
  ./src/sjev eval "examples/work/compare_t$t.sjev" examples/work/test.jsonl
done
