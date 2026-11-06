#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-$(cat ../../../.bazelversion)}"

# Keep the releases pinned even in CI jobs testing a different CUDA version.
unset CUDA_REDIST_VERSION_OVERRIDE CUDA_REDIST_SHA256_OVERRIDE CUDA_REDIST_INTEGRITY_OVERRIDE

for version in 11.0.3 11.8.0 13.3.1; do
    args=(--enable_bzlmod --@rules_cuda//cuda:version="$version" --@rules_cuda//cuda:compiler=nvcc --@rules_cuda//cuda:enable=True)

    # Loading and analysis succeed even for mandatory consumers of missing parts.
    bazel query 'deps(//:disabled_component_test)' --enable_bzlmod >/dev/null
    bazel build //:required_component //:required_compiler --nobuild "${args[@]}"

    # CCCL is empty in 11.0, present under cuda_cccl in 11.8, and cccl in 13.3.
    headers=$(bazel cquery @cuda//:cccl_headers "${args[@]}" --output=starlark \
        --starlark:expr='len([v for k, v in providers(target).items() if k.endswith("CcInfo")][0].compilation_context.headers.to_list())')
    if [[ "$version" == 11.0.3 ]]; then
        [[ "$headers" == 0 ]]
    else
        [[ "$headers" =~ ^[1-9][0-9]*$ ]]
    fi

    # Consumers can handle absence in their code or omit the dependency entirely.
    bazel test //:optional_component_test "${args[@]}"
    bazel test //:disabled_component_test "${args[@]}" --@rules_cuda//cuda:enable=False

    # Mandatory use fails during an action, after the successful analysis above.
    if output=$(bazel build //:required_component "${args[@]}" 2>&1); then
        echo "$output" >&2
        echo "Expected compilation to fail without nvJitLink" >&2
        exit 1
    fi
    if [[ "$output" != *"nvJitLink.h"* ]]; then
        echo "$output" >&2
        exit 1
    fi
    if output=$(bazel build //:required_compiler "${args[@]}" 2>&1); then
        echo "$output" >&2
        echo "Expected compilation to fail without nvcc" >&2
        exit 1
    fi
    if [[ "$output" != *"nvcc of cuda toolkit does not exist"* ]]; then
        echo "$output" >&2
        exit 1
    fi
done

# Do not leave execution-root symlinks for the parent workspace's //tests/... scan.
bazel clean
bazel shutdown
