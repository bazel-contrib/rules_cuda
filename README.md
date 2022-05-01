# CUDA Rules for [Bazel](https://bazel.build)

**WARNING**: WIP, expect breakage!

This repository contains pure [Starlark](https://github.com/bazelbuild/starlark) implementation of CUDA rules. These
rules provide some macros and rules that make it easier to build CUDA with Bazel.

## Reference documentation

### Workspace setup

Paste the following snippet into your `WORKSPACE` file and replace the placeholders to actual values.

```py
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    sha256 = "{sha256_to_replace}",
    strip_prefix = "rules_cuda-{git_commit_hash}",
    urls = ["https://github.com/cloudhan/rules_cuda/archive/{git_commit_hash}.tar.gz"],
)

load("@rules_cuda//cuda:deps.bzl", "register_detected_cuda_toolchains", "rules_cuda_deps")

rules_cuda_deps()

register_detected_cuda_toolchains()
```

**NOTE**: the use of `register_detected_cuda_toolchains` depends on the environment variable `CUDA_PATH`. You must also
ensure the host compile is available. On windows, this means you will also need to set the environment variable
`BAZEL_VC` properly.

### Rules

- `cuda_library`: Can be used to compile and create static library for CUDA kernel code. The resulting targets can be
  consumed by [C/C++ Rules](https://bazel.build/reference/be/c-cpp#rules).
- `cuda_objects`: If you don't understand what *device link* means, you must never use it. This rule produce incomplete
  object files that can only be consumed by `cuda_library`. It is created for relocatable device code and device link
  time optimization source files.

### Flags

Some flags are defined in [cuda/BUILD.bazel](cuda/BUILD.bazel). To use them, for example:
```
bazel build --@rules_cuda//cuda:archs=compute_61:compute_61,sm_61
```

In `.bazelrc` file, you can define shortcut alias for the flag, for example:
```
# Convenient flag shortcuts.
build --flag_alias=cuda_archs=@rules_cuda//cuda:archs
```
and then you can use it as following:
```
bazel build --cuda_archs=compute_61:compute_61,sm_61
```

## Examples

See [examples](./examples) for basic usage.

See [this examples repo](https://github.com/cloudhan/rules_cuda_examples) for extended real world usage.
