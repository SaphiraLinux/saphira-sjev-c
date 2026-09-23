#!/bin/sh
# Evaluate a trained model on the synthetic test split. Usage:
#   sh examples/eval_synthetic.sh [model]
set -eu
cd "$(dirname "$0")/.."
MODEL="${1:-examples/work/synthetic.sjev}"
if [ ! -f examples/work/test.jsonl ]; then
  echo "missing examples/work/test.jsonl (run examples/train_synthetic.sh first)" >&2
  exit 1
fi
./src/sjev eval "$MODEL" examples/work/test.jsonl
