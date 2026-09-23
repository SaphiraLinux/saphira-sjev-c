#!/bin/sh
# Sanitizer legs: rebuild under asan/ubsan/tsan, run a small threaded
# train+predict+eval smoke under each, then restore the normal binary.
# Serial by construction (sanitizer targets are never parallel with clean).
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
W=$(mkwork)
smoke() {
  tag="$1"
  "$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
    --output "$W/m.sjev" --width 16 --rank 16 --context-tokens 64 \
    --option-tokens 32 --epochs 2 --batch-size 8 --seed 7 --threads 4 \
    >"$W/train_$tag.log" 2>&1 || fail "$tag train failed"
  "$SJEV" predict "$W/m.sjev" --context "Pick the badge: gold heron." \
    --option "azure crane" --option "gold heron" >"$W/p_$tag.json" 2>/dev/null \
    || fail "$tag predict failed"
  "$SJEV" eval "$W/m.sjev" tests/data/test.jsonl >"$W/e_$tag.json" 2>/dev/null \
    || fail "$tag eval failed"
}
make -C src asan >/dev/null 2>&1 || fail "asan build failed"
smoke asan
make -C src ubsan >/dev/null 2>&1 || fail "ubsan build failed"
smoke ubsan
make -C src tsan >/dev/null 2>&1 || fail "tsan build failed"
smoke tsan
make -C src clean >/dev/null 2>&1
make -C src >/dev/null 2>&1 || fail "normal rebuild failed"
need_bin
rm -rf "$W"
pass "sanitizers"
