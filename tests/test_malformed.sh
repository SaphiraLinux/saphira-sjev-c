#!/bin/sh
# Corrupt/truncated model files must fail cleanly, never crash.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
W=$(mkwork)
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/m.sjev" --width 16 --rank 16 --context-tokens 64 \
  --option-tokens 32 --epochs 1 --batch-size 8 --seed 7 >/dev/null 2>&1 \
  || fail "train failed"
head -c 20 "$W/m.sjev" > "$W/trunc.sjev"
printf 'BAD!' > "$W/badmagic.sjev"
head -c 100 /dev/urandom > "$W/garbage.sjev"
: > "$W/empty.sjev"
for f in trunc badmagic garbage empty; do
  if "$SJEV" predict "$W/$f.sjev" --context "hi" \
      --option "a" --option "b" >/dev/null 2>&1; then
    fail "predict accepted $f"
  fi
  if "$SJEV" eval "$W/$f.sjev" tests/data/test.jsonl >/dev/null 2>&1; then
    fail "eval accepted $f"
  fi
done
rm -rf "$W"
pass "malformed"
