#!/bin/sh
# CLI help and error paths exit with the right status and mention --threads.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
need_bin
"$SJEV" --help >/dev/null 2>&1 || fail "--help failed"
"$SJEV" predict --help 2>&1 | grep -q "predict" || fail "predict help missing"
"$SJEV" train --help 2>&1 | grep -q -- "--threads" || fail "train help lacks --threads"
if "$SJEV" frobnicate >/dev/null 2>&1; then fail "unknown command accepted"; fi
if "$SJEV" predict >/dev/null 2>&1; then fail "predict without args accepted"; fi
if "$SJEV" train >/dev/null 2>&1; then fail "train without args accepted"; fi
if "$SJEV" eval >/dev/null 2>&1; then fail "eval without args accepted"; fi
if "$SJEV" train --threads 0 /tmp/nope.jsonl >/dev/null 2>&1; then fail "--threads 0 accepted"; fi
if "$SJEV" train --threads abc /tmp/nope.jsonl >/dev/null 2>&1; then fail "--threads abc accepted"; fi
if "$SJEV" train --threads 1025 /tmp/nope.jsonl >/dev/null 2>&1; then fail "--threads 1025 accepted"; fi
if "$SJEV" eval model.sjev test.jsonl --threads 2 >/dev/null 2>&1; then fail "eval accepted --threads"; fi
if "$SJEV" grad-check >/dev/null 2>&1; then fail "grad-check without args accepted"; fi
pass "cli"
