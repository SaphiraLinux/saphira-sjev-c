#!/bin/sh
# Cross-branch parity: build Master (single-thread baseline) in a temporary
# git worktree, train/evaluate both from identical seeds and compare metrics
# within FP32 tolerance. No Jevlike/PyTorch oracle involved.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
W=$(mkwork)
git worktree add --detach "$W/master" origin/Master >/dev/null 2>&1 \
  || fail "worktree add failed"
trap 'git worktree remove --force "$W/master" >/dev/null 2>&1 || true' EXIT INT TERM
# Master is flat-layout: sources at worktree root.
make -C "$W/master" >/dev/null 2>&1 || fail "master build failed"
[ -x "$W/master/sjev" ] || fail "master binary missing"
need_bin
common="--width 16 --rank 16 --context-tokens 64 --option-tokens 32 --epochs 3 --batch-size 8 --seed 7"
# shellcheck disable=SC2086
"$W/master/sjev" train tests/data/train.jsonl \
  --validation tests/data/validation.jsonl \
  --output "$W/m_master.sjev" $common >"$W/master.log" 2>&1 \
  || fail "master train failed"
# shellcheck disable=SC2086
"$SJEV" train tests/data/train.jsonl --validation tests/data/validation.jsonl \
  --output "$W/m_mt.sjev" $common --threads 1 >"$W/mt.log" 2>&1 \
  || fail "multithread train failed"
"$W/master/sjev" eval "$W/m_master.sjev" tests/data/test.jsonl >"$W/e_master.json" 2>/dev/null \
  || fail "master eval failed"
"$SJEV" eval "$W/m_mt.sjev" tests/data/test.jsonl >"$W/e_mt.json" 2>/dev/null \
  || fail "multithread eval failed"
python3 - "$W/e_master.json" "$W/e_mt.json" << 'EOF' || fail "master parity diverged"
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for key in ("test_nll", "top1", "top3"):
    d = abs(a[key] - b[key])
    assert d < 1e-4, (key, a[key], b[key], d)
EOF
git worktree remove --force "$W/master" >/dev/null 2>&1
trap - EXIT INT TERM
rm -rf "$W"
pass "master_parity"
