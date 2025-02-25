#!/bin/bash

this_dir=$(realpath $(dirname $0))

set -ex

# toolchain configured by the root module of the user
pushd "$this_dir/toolchain_root"
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True
    bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=False
    bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=True
    bazel build //:use_library
    bazel build //:use_rule
    bazel clean && bazel shutdown
popd

# toolchain does not exists
pushd "$this_dir/toolchain_none"
    # analysis pass
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True

    # force build optional targets
    bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=False
    ERR=$(bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=True 2>&1 || true)
    if ! [[ $ERR == *"didn't satisfy constraint"*"valid_toolchain_is_configured"* ]]; then exit 1; fi

    # use library fails because the library file does not exist
    ERR=$(bazel build //:use_library 2>&1 || true)
    if ! [[ $ERR =~ "target 'cuda_runtime' not declared in package" ]]; then exit 1; fi
    if ! [[ $ERR =~ "ERROR: Analysis of target '//:use_library' failed" ]]; then exit 1; fi

    # use rule fails because rules_cuda depends non-existent cuda toolkit
    ERR=$(bazel build //:use_rule 2>&1 || true)
    if ! [[ $ERR =~ "target 'cuda_runtime' not declared in package" ]]; then exit 1; fi
    if ! [[ $ERR =~ "ERROR: Analysis of target '//:use_rule' failed" ]]; then exit 1; fi

    bazel clean && bazel shutdown
popd

# toolchain configured by rules_cuda
pushd "$this_dir/toolchain_rules"
    bazel build //... --@rules_cuda//cuda:enable=False
    bazel build //... --@rules_cuda//cuda:enable=True
    bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=False
    bazel build //:optinally_use_rule --@rules_cuda//cuda:enable=True
    bazel build //:use_library
    bazel build //:use_rule
    bazel clean && bazel shutdown
popd

# toolchain configured with deliverables (manual components with workspace)
pushd "$this_dir/toolchain_components"
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:optinally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //:optinally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:use_library
    bazel build --enable_workspace //:use_rule
    bazel clean && bazel shutdown
popd

# toolchain configured with deliverables (manual components with bzlmod)
pushd "$this_dir/toolchain_components"
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=False
    bazel build --enable_bzlmod //... --@rules_cuda//cuda:enable=True
    bazel build --enable_bzlmod //:optinally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_bzlmod //:optinally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_bzlmod //:use_library
    bazel build --enable_bzlmod //:use_rule
    bazel clean && bazel shutdown
popd

# toolchain configured with deliverables (redistrib.json with workspace)
pushd "$this_dir/toolchain_redist_json"
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //... --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:optinally_use_rule --@rules_cuda//cuda:enable=False
    bazel build --enable_workspace //:optinally_use_rule --@rules_cuda//cuda:enable=True
    bazel build --enable_workspace //:use_library
    bazel build --enable_workspace //:use_rule
    bazel clean && bazel shutdown
popd
