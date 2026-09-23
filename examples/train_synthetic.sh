#!/bin/sh
# Train on generated synthetic data. Usage:
#   sh examples/train_synthetic.sh [output-model] [threads]
set -eu
cd "$(dirname "$0")/.."
MODEL="${1:-examples/work/synthetic.sjev}"
THREADS="${2:-1}"
THREADS=$((${THREADS:-1} + 0))
[ -d examples/work ] || python3 tools/make_synthetic.py \
  --output-dir examples/work --train 2000 --validation 400 --test 400 --seed 7
./src/sjev train examples/work/train.jsonl \
  --validation examples/work/validation.jsonl \
  --output "$MODEL" \
  --width 64 --rank 64 --context-tokens 192 --option-tokens 32 \
  --epochs 8 --batch-size 64 --seed 7 --threads "$THREADS"
