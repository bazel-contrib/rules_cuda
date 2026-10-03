#!/bin/bash

this_dir=$(realpath $(dirname $0))

# Parse arguments
skip_root=false
skip_none=false
skip_rules=false
skip_components_workspace=false
skip_components_bzlmod=false
skip_redist_json=false
skip_redist_json_multi=false
skip_redist_json_collision=false
skip_redist_json_version_gate=false
skip_cccl_headers=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-root)
            skip_root=true; shift ;;
        --no-none)
            skip_none=true; shift ;;
        --no-rules)
            skip_rules=true; shift ;;
        --no-components-workspace)
            skip_components_workspace=true; shift ;;
        --no-components-bzlmod)
            skip_components_bzlmod=true; shift ;;
        --no-components)
            skip_components_workspace=true; skip_components_bzlmod=true; shift ;;
        --no-redist)
            skip_redist_json=true; shift ;;
        --no-redist-multi)
            skip_redist_json_multi=true; shift ;;
        --no-redist-collision)
            skip_redist_json_collision=true; shift ;;
        --no-redist-version-gate)
            skip_redist_json_version_gate=true; shift ;;
        --no-cccl-headers)
            skip_cccl_headers=true; shift ;;
        *)
            echo "Unknown option: $1" >&2; shift ;;
    esac
done

set -ex

redist_platform_args=()
if [[ "$RUNNER_OS" == "Windows" ]] || [[ "$(uname -s 2>/dev/null)" =~ MINGW|MSYS|CYGWIN ]]; then
    # Target redist is auto-detected from the Windows host platform's
    # @platforms//os:windows + @platforms//cpu:x86_64 constraints. exec_platform
    # is set defensively to guarantee the windows-x86_64 nvcc redist regardless
    # of how the exec configuration's platform is declared.
    redist_platform_args=(
        --@rules_cuda//cuda:exec_platform=windows-x86_64
    )
fi

# toolchain configured by the root module of the user
if [ "$skip_root" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN CONFIGURED BY ROOT MODULE
============================================================
EOF
pushd "$this_dir/toolchain_root"
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True
    bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=False
    bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=True
    bazel build //:use_library
    bazel build //:use_rule
    bazel clean && bazel shutdown
popd
fi

# conflicting redistrib.json definitions should fail during module extension evaluation
if [ "$skip_redist_json_collision" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH REDISTRIB.JSON CONFLICT (BZLMOD)
============================================================
EOF
pushd "$this_dir/toolchain_redist_json_collision"
    ERR=$(CUDA_REDIST_VERSION_OVERRIDE= bazel build --enable_bzlmod //:probe 2>&1 || true)
    if ! [[ $ERR == *"Conflicting CUDA component definition for cudart on linux-x86_64 at version"* ]]; then exit 1; fi
    bazel clean && bazel shutdown
popd
fi

# toolchain does not exists
if [ "$skip_none" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN DOES NOT EXIST
============================================================
EOF
pushd "$this_dir/toolchain_none"
    # analysis pass
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True

    # force build optional targets
    bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=False
    ERR=$(bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=True 2>&1 || true)
    if ! [[ $ERR == *"didn't satisfy constraint"*"valid_toolchain_is_configured"* ]]; then exit 1; fi

    # use library should analyse build successfully (empty cuda_runtime target exists)
    bazel build //:use_library

    # use rule analyses correctly but fails during compilation because cuda toolkit doesn't exist
    ERR=$(bazel build //:use_rule 2>&1 || true)
    # nvcc toolchain fails with "nvcc of cuda toolkit does not exist", clang toolchain fails with "cannot find CUDA installation"
    if ! [[ $ERR =~ "nvcc of cuda toolkit does not exist" || $ERR =~ "cannot find CUDA installation" ]]; then exit 1; fi
    if ! [[ $ERR =~ "ERROR: Build did NOT complete successfully" ]]; then exit 1; fi

    bazel clean && bazel shutdown
popd
fi

# toolchain configured by rules_cuda
if [ "$skip_rules" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN CONFIGURED BY RULES_CUDA
============================================================
EOF
pushd "$this_dir/toolchain_rules"
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True
    bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=False
    bazel build //:optionally_use_rule --@rules_cuda//cuda:enable=True
    bazel build //:use_library
    bazel build //:use_rule
    bazel clean && bazel shutdown
popd
fi

# toolchain configured with deliverables (manual components with workspace)
if [ "$skip_components_workspace" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH MANUAL COMPONENTS (WORKSPACE)
============================================================
EOF
pushd "$this_dir/toolchain_components"
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:use_library
    bazel build --enable_workspace //:use_rule
    bazel clean && bazel shutdown
popd
fi

# toolchain configured with deliverables (manual components with bzlmod)
if [ "$skip_components_bzlmod" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH MANUAL COMPONENTS (BZLMOD)
============================================================
EOF
pushd "$this_dir/toolchain_components"
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=False
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=True
    bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_bzlmod //:use_library
    bazel build --enable_bzlmod //:use_rule
    bazel clean && bazel shutdown
popd
fi

# toolchain configured with deliverables (redistrib.json with workspace)
if [ "$skip_redist_json" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH REDISTRIB.JSON (WORKSPACE)
============================================================
EOF
pushd "$this_dir/toolchain_redist_json"
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=False "${redist_platform_args[@]}"
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=True "${redist_platform_args[@]}"
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=False "${redist_platform_args[@]}"
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=True "${redist_platform_args[@]}"
    bazel build --enable_workspace //:use_library "${redist_platform_args[@]}"
    bazel build --enable_workspace //:use_rule "${redist_platform_args[@]}"
    bazel clean && bazel shutdown
popd
fi

# toolchain configured with redistrib.json (multi-version with bzlmod)
if [ "$skip_redist_json_multi" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH REDISTRIB.JSON (BZLMOD MULTI-VERSION)
============================================================
EOF
pushd "$this_dir/toolchain_redist_json_multi"
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=False "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=True "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=False "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=True --@rules_cuda//cuda:version=12.6.3 "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=True --@rules_cuda//cuda:version=11.7.0 "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:use_library "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:use_rule --@rules_cuda//cuda:version=12.6.3 "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:use_rule --@rules_cuda//cuda:version=11.7.0 "${redist_platform_args[@]}"

    # Keep the override-only dedupe probe isolated so it cannot pollute later versioned builds.
    bazel clean && bazel shutdown
    CUDA_REDIST_VERSION_OVERRIDE=11.7.0 bazel build --enable_bzlmod //:optionally_use_rule --@rules_cuda//cuda:enable=False "${redist_platform_args[@]}"
    bazel clean && bazel shutdown
popd
fi

# Compiling against the LOWER of two declared versions, where the two straddle a
# version-gated nvcc flag. The toolchain used to report the maximum declared version
# regardless of which nvcc the component aliases resolved to, so a 12.x compile was
# handed a 12.9+-only flag and nvcc rejected it.
#
# Two things are pinned rather than inherited from the environment, because both would
# otherwise stop this testing what it is here to test:
#
#   - CUDA_REDIST_VERSION_OVERRIDE is unset. CI exports it for the whole script, and it
#     rewrites the version of EVERY redist_json, collapsing both declarations onto one
#     version.
#   - The compiler is pinned to nvcc. The flag under test is an nvcc flag, and clang
#     brings its own version coupling: it targets a PTX ISA its own release chose, which
#     the older toolkit's ptxas then rejects ("Unsupported .version 8.8").
if [ "$skip_redist_json_version_gate" = false ]; then
cat <<- EOF

============================================================
=== TEST: TOOLCHAIN WITH REDISTRIB.JSON (BZLMOD VERSION GATE)
============================================================
EOF
pushd "$this_dir/toolchain_redist_json_version_gate"
    version_gate_args=(--@rules_cuda//cuda:compiler=nvcc "${redist_platform_args[@]}")
    env -u CUDA_REDIST_VERSION_OVERRIDE bazel build --enable_bzlmod //:kernel_lib --@rules_cuda//cuda:version=12.9.1 "${version_gate_args[@]}"
    env -u CUDA_REDIST_VERSION_OVERRIDE bazel build --enable_bzlmod //:kernel_lib --@rules_cuda//cuda:version=12.8.1 "${version_gate_args[@]}"
    bazel clean && bazel shutdown
popd
fi

# `<thrust/...>` resolving through the header-only targets. CUDA 13 moved the CCCL headers
# into `include/cccl/`, so a version-unaware `includes` resolves `<cccl/thrust/...>` and
# these stop compiling. One declared version, so CUDA_REDIST_VERSION_OVERRIDE sweeps this
# across the CUDA versions in the CI matrix and covers both layouts.
if [ "$skip_cccl_headers" = false ]; then
cat <<- EOF

============================================================
=== TEST: CCCL HEADER-ONLY TARGETS (BZLMOD)
============================================================
EOF
pushd "$this_dir/toolchain_cccl_headers"
    bazel build --enable_bzlmod //:via_cccl_headers "${redist_platform_args[@]}"
    bazel build --enable_bzlmod //:via_cuda_headers "${redist_platform_args[@]}"
    bazel clean && bazel shutdown
popd
fi
