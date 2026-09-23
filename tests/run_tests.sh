#!/bin/sh
# SJEV functional suite. Run from the repository root: sh tests/run_tests.sh
# or via: make check
set -eu
cd "$(dirname "$0")/.."
pass=0
failed=0
for t in tests/test_*.sh; do
    case "$t" in
        tests/run_tests.sh) continue;;
    esac
    if sh "$t"; then
        pass=$((pass + 1))
    else
        echo "SUITE FAIL: $t" >&2
        failed=$((failed + 1))
    fi
done
echo "suite: $pass passed, $failed failed"
[ "$failed" -eq 0 ]
