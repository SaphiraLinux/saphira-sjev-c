#!/bin/sh
# Build from clean works and yields an executable binary.
set -eu
cd "$(dirname "$0")/.."
. tests/lib.sh
make -C src clean >/dev/null 2>&1
make -C src >/dev/null 2>&1 || fail "make failed"
[ -x src/sjev ] || fail "src/sjev not built"
pass "build"
