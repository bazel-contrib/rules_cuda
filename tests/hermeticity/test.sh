#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

set -euo pipefail

# Parse command-line flags
ALLOW_NON_HERMETIC=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --allow-non-hermetic)
            ALLOW_NON_HERMETIC=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--allow-non-hermetic]"
            exit 1
            ;;
    esac
done

file=$(bazel info execution_root)/$(bazel cquery --output=files :cuda_test)
hermetic_flags=(\
    --@rules_cuda//cuda:copts=-Xcompiler=-ffile-prefix-map=$(pwd)=. \
    --@rules_cuda//cuda:copts=-Xcompiler=-fdebug-prefix-map=$(pwd)=. \
    --@rules_cuda//cuda:copts=-objtemp)

bazel clean

bazel build "${hermetic_flags[@]}" :cuda_test
build_output1=$(strings ${file})

bazel clean

bazel build "${hermetic_flags[@]}" :cuda_test
build_output2=$(strings ${file})

diff_output=$(diff -u <(echo "$build_output1") <(echo "$build_output2") --color=always || true)

if [[ -n "$diff_output" ]]; then
    if [[ "$ALLOW_NON_HERMETIC" == "1" ]]; then
        echo -e "${YELLOW}WARNING: Build outputs are not hermetic (differences detected), but continuing due to --allow-non-hermetic flag:${NC}"
        echo "$diff_output"
        echo -e "${YELLOW}Note: This is expected, hermeticity is only guaranteed with some newer nvcc versions${NC}"
        exit 0
    else
        echo -e "${RED}ERROR: Build outputs are not hermetic (differences detected):${NC}"
        echo "$diff_output"
        exit 1
    fi
else
    echo -e "${GREEN}SUCCESS: Build outputs are hermetic (no differences)${NC}"
fi
