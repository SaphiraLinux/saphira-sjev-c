#!/bin/sh
# Train on the fixture: loss must decrease; eval must report metrics.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
W=$(mkwork)
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/m.sjev" --width 32 --rank 32 --context-tokens 128 \
  --option-tokens 64 --epochs 6 --batch-size 16 --seed 7 >"$W/train.log" 2>&1 \
  || fail "train failed"
python3 - "$W/train.log" << 'EOF' || fail "loss did not decrease"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"epoch"' in l]
first = rows[0]["train_nll"]
last = rows[-1]["train_nll"]
assert last < first * 0.85, (first, last)
assert rows[-1]["val_top1"] > 0.30, rows[-1]
EOF
"$SJEV" eval "$W/m.sjev" tests/data/test.jsonl >"$W/eval.json" 2>/dev/null \
  || fail "eval failed"
python3 - "$W/eval.json" << 'EOF' || fail "eval metrics bad"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["examples"] == 32, d
assert d["top1"] > 0.30, d
assert 0.0 <= d["ece"] <= 1.0, d
EOF
rm -rf "$W"
pass "train_eval"
