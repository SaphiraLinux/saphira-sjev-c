#!/bin/sh
# --threads 1 vs 2/4/8: same seed, metrics within FP32 noise; models near-identical.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
W=$(mkwork)
base="--width 16 --rank 16 --context-tokens 64 --option-tokens 32 --epochs 3 --batch-size 8 --seed 7"
# shellcheck disable=SC2086
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/t1.sjev" $base --threads 1 >"$W/t1.log" 2>&1 || fail "threads 1 failed"
for t in 2 4 8; do
  # shellcheck disable=SC2086
  "$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
    --output "$W/t$t.sjev" $base --threads "$t" >"$W/t$t.log" 2>&1 \
    || fail "threads $t failed"
  python3 - "$W/t1.log" "$W/t$t.log" "$W/t1.sjev" "$W/t$t.sjev" << 'EOF' \
    || fail "threads 1 vs $t diverged"
import json, struct, sys

def last(path, key):
    rows = [json.loads(l) for l in open(path) if '"epoch"' in l]
    return rows[-1][key]

_, l1, lt, m1, mt = sys.argv
for key in ("train_nll", "validation_nll", "val_top1"):
    d = abs(last(l1, key) - last(lt, key))
    assert d < 5e-4, (key, d)

def floats(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    assert blob[:4] == b"SJEV", blob[:4]
    n = (len(blob) - 32) // 4
    return struct.unpack(f"<{n}f", blob[32:])

a, b = floats(m1), floats(mt)
assert len(a) == len(b)
print("maxabs", max(abs(x - y) for x, y in zip(a, b)))
assert max(abs(x - y) for x, y in zip(a, b)) < 1e-3, "model weights diverged"
EOF
done
rm -rf "$W"
pass "threads"
