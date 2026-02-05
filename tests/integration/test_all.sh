#!/bin/bash

this_dir=$(realpath $(dirname $0))

# Parse arguments
skip_root=false
skip_none=false
skip_rules=false
skip_components_workspace=false
skip_components_bzlmod=false
skip_redist_json=false

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
        *)
            echo "Unknown option: $1" >&2; shift ;;
    esac
done

set -ex

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
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //:optionally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:use_library
    bazel build --enable_workspace //:use_rule
    bazel clean && bazel shutdown
popd
fi
