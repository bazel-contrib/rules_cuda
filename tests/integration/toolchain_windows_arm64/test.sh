#!/bin/bash
# Check Windows ARM64 repository generation and selection on any host, without MSVC.
set -euo pipefail
cd "$(dirname "$0")"
export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-$(cat ../../../.bazelversion)}"
unset CUDA_REDIST_VERSION_OVERRIDE CUDA_REDIST_SHA256_OVERRIDE CUDA_REDIST_INTEGRITY_OVERRIDE

args=(--enable_bzlmod --platforms=//:windows_arm64)
for version in '' 13.4.2; do
    # Target components and automatic compiler selection must use ARM64 archives.
    for target in cudart_all_files nvcc cicc; do
        files=$(bazel cquery "@cuda//:${target}" "${args[@]}" \
            --@rules_cuda//cuda:version="$version" --output=files)
        [[ "$files" == *windows_arm64* ]]
        [[ "$files" != *windows_x86_64* ]]
        case "$target" in
            cudart_all_files) [[ "$files" == *lib/arm64/cudart.lib* ]] ;;
            nvcc) [[ "$files" == *bin/nvcc.exe* ]] ;;
            cicc) [[ "$files" == *bin/cicc.exe* ]] ;;
        esac
    done
done

# The explicit execution-platform override must also accept windows-arm64.
files=$(bazel cquery @cuda//:nvcc --enable_bzlmod \
    --@rules_cuda//cuda:exec_platform=windows-arm64 --output=files)
[[ "$files" == *windows_arm64*bin/nvcc.exe* ]]

# Follow the public aliases to inspect the ARM64 component's import libraries.
# Unconfigured query also visits x86_64 branches, so select the ARM64 results.
for target in cuda_runtime cudadevrt_lib cuda; do
    library=$(bazel query "filter(windows_arm64, labels(interface_library, deps(@cuda//:${target})))" --enable_bzlmod)
    [[ "$library" == *lib/arm64/* ]]
    [[ "$library" != *lib/x64/* ]]
done

# Check default and versioned Windows registrations. Neither may require
# x86_64 on either the execution or target platform.
for target in \
    @cuda//toolchain:nvcc-windows-toolchain \
    @cuda//toolchain:nvcc-windows-13_4_2-toolchain; do
    declaration=$(bazel query "$target" --enable_bzlmod --output=build)
    [[ "$declaration" == *os:windows* ]]
    [[ "$declaration" != *cpu:x86_64* ]]
done

# Discover the underlying registration beside the implementation reached through
# the public facade, without importing an implementation repository by name.
declaration=$(bazel query 'kind("^toolchain rule$", siblings(kind("^cuda_toolchain rule$", deps(labels(toolchain, @cuda//toolchain:nvcc-windows-toolchain), 1))))' \
    --enable_bzlmod --output=build)
[[ "$declaration" == *os:windows* ]]
[[ "$declaration" != *cpu:x86_64* ]]

# Avoid leaving execution-root symlinks in the parent workspace's //tests/... scan.
bazel clean
bazel shutdown
