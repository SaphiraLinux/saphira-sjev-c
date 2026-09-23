#!/bin/sh
# Train a tiny model, check predict determinism and probability sum = 1.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
W=$(mkwork)
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/m.sjev" --width 16 --rank 16 --context-tokens 64 \
  --option-tokens 32 --epochs 2 --batch-size 16 --seed 7 >"$W/train.log" 2>&1 \
  || fail "train failed"
[ -s "$W/m.sjev" ] || fail "no model written"
"$SJEV" predict "$W/m.sjev" --context "Pick the badge: gold heron." \
  --option "azure crane" --option "gold heron" --option "bronze ibis" \
  >"$W/p1.json" 2>/dev/null || fail "predict failed"
"$SJEV" predict "$W/m.sjev" --context "Pick the badge: gold heron." \
  --option "azure crane" --option "gold heron" --option "bronze ibis" \
  >"$W/p2.json" 2>/dev/null || fail "predict rerun failed"
cmp "$W/p1.json" "$W/p2.json" || fail "predict not deterministic"
python3 - "$W/p1.json" << 'EOF' || fail "probability sum != 1"
import json, sys
rows = json.load(open(sys.argv[1]))
s = sum(r["probability"] for r in rows)
assert abs(s - 1.0) < 1e-6, s
assert all(r["logit"] == r["logit"] for r in rows)
EOF
rm -rf "$W"
pass "init_predict"
