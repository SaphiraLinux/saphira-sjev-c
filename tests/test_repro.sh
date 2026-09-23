#!/bin/sh
# Fixed-seed reproducibility: same seed twice -> byte-identical model.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
W=$(mkwork)
common="--width 16 --rank 16 --context-tokens 64 --option-tokens 32 --epochs 2 --batch-size 8 --seed 7"
# shellcheck disable=SC2086
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/a.sjev" $common >/dev/null 2>&1 || fail "train a failed"
# shellcheck disable=SC2086
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/b.sjev" $common >/dev/null 2>&1 || fail "train b failed"
cmp "$W/a.sjev" "$W/b.sjev" || fail "same-seed models differ"
rm -rf "$W"
pass "repro"
