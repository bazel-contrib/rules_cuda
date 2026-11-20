#!/usr/bin/env bash
set -euo pipefail

source "$TEST_SRCDIR/bazel_tools/tools/bash/runfiles/runfiles.bash"
launcher="$(rlocation '%{launcher}')"
args=(--tool initcheck -- '/test path/program' 'argument with spaces' '' '$literal')
expected=(--error-exitcode 1 --target-processes all "${args[@]}")

check_args() {
  local actual=() arg
  while IFS= read -r -d '' arg; do
    actual+=("$arg")
  done < "$TEST_TMPDIR/sanitizer_args"
  test "${#actual[@]}" = "${#expected[@]}"
  for i in "${!expected[@]}"; do
    test "${actual[$i]}" = "${expected[$i]}"
  done
}

"$launcher" "${args[@]}"
check_args

# Preserve sanitizer findings and application failures as nonzero test exits.
for code in 1 42; do
  status=0
  FAKE_SANITIZER_EXIT_CODE="$code" "$launcher" "${args[@]}" || status=$?
  test "$status" = "$code"
done

# A run_under launcher must also work without its own adjacent runfiles tree.
cp "$launcher" "$TEST_TMPDIR/launcher"
RUNFILES_DIR="$TEST_SRCDIR" "$TEST_TMPDIR/launcher" "${args[@]}"
check_args

# Exercise manifest-only lookup independently of the launcher's runfiles tree.
manifest="$TEST_TMPDIR/MANIFEST"
printf '%s %s\n' \
  bazel_tools/tools/bash/runfiles/runfiles.bash \
  "$TEST_SRCDIR/bazel_tools/tools/bash/runfiles/runfiles.bash" \
  "$TEST_WORKSPACE/tests/compute_sanitizer/fake_sanitizer.sh" \
  "$TEST_SRCDIR/$TEST_WORKSPACE/tests/compute_sanitizer/fake_sanitizer.sh" > "$manifest"
RUNFILES_DIR='' RUNFILES_MANIFEST_FILE="$manifest" RUNFILES_MANIFEST_ONLY=1 \
  "$TEST_TMPDIR/launcher" "${args[@]}"
check_args
