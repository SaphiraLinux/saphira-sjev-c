#!/bin/sh
# SJEV quickstart from a fresh clone: build, generate data, train,
# evaluate, predict. Ends with a trained model and its metrics.
set -eu
cd "$(dirname "$0")/.."
make >/dev/null 2>&1
python3 tools/make_synthetic.py --output-dir examples/work \
  --train 800 --validation 150 --test 150 --seed 7
./src/sjev train examples/work/train.jsonl \
  --validation examples/work/validation.jsonl \
  --output examples/work/model.sjev \
  --width 32 --rank 32 --context-tokens 128 --option-tokens 64 \
  --epochs 6 --batch-size 32 --seed 7
./src/sjev eval examples/work/model.sjev examples/work/test.jsonl
./src/sjev predict examples/work/model.sjev \
  --context "Pick the badge: gold heron." \
  --option "azure crane" \
  --option "gold heron" \
  --option "bronze ibis"
