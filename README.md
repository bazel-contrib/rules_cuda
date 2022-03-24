# CUDA Rules for [Bazel](https://bazel.build)

**WARNING**: WIP, expect breakage!

This repository contains pure [Starlark](https://github.com/bazelbuild/starlark) implementation of CUDA rules. These
rules provide some macros and rules that make it easier to build CUDA with Bazel.

## Reference documentation

- `cuda_library`: Can be used to compile and create static library for CUDA kernel code. The resulting targets can be
  consumed by [C/C++ Rules](https://bazel.build/reference/be/c-cpp#rules).
- `cuda_objects`: If you don't understand what *device link* means, you must never use it. This rule produce incomplete
  object files that can only be consumed by `cuda_library`. It is created for relocatable device code and device link
  time optimization source files.

## Examples

See [examples](./examples).
