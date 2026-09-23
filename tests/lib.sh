#!/bin/sh
# Shared helpers for the SJEV shell test suite. Source, do not execute.
# Tests run from the repository root (see tests/run_tests.sh).
SJEV="${SJEV_BIN:-src/sjev}"

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

pass() {
    echo "PASS: $1"
}

need_bin() {
    [ -x "$SJEV" ] || fail "binary $SJEV missing (run make first)"
}

mkwork() {
    _d=$(mktemp -d "${TMPDIR:-/tmp}/sjev-test-XXXXXX") || fail "mktemp failed"
    echo "$_d"
}

# usage: assert_contains FILE SUBSTRING
assert_contains() {
    grep -qF "$2" "$1" || fail "expected '$2' in $1"
}
