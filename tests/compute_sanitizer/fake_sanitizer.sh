#!/usr/bin/env bash
set -euo pipefail

# Model the sanitizer's dependency on files adjacent to its executable.
test "$(cat "$(dirname "$0")/support.txt")" = "sanitizer support file"
printf '%s\0' "$@" > "$TEST_TMPDIR/sanitizer_args"
exit "${FAKE_SANITIZER_EXIT_CODE:-0}"
