#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

set -euo pipefail

file=$(bazel info execution_root)/$(bazel cquery --output=files :cuda_test)

bazel build :cuda_test
build_output1=$(strings ${file})

bazel clean

bazel build :cuda_test
build_output2=$(strings ${file})

diff_output=$(diff -u <(echo "$build_output1") <(echo "$build_output2") --color=always || true)

if [[ -n "$diff_output" ]]; then
    echo -e "${RED}ERROR: Build outputs are not hermetic (differences detected):${NC}"
    echo "$diff_output"
    exit 1
else
    echo -e "${GREEN}SUCCESS: Build outputs are hermetic (no differences)${NC}"
fi
