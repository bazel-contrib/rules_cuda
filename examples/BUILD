load("//cuda:defs.bzl", "cuda_library", "requires_cuda_enabled")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
    target_compatible_with = requires_cuda_enabled(),
)

# Run with 'bazel run --cuda //examples:main_kernel'
# NOTE: --cuda is added via flag_alias in .bazelrc
cc_binary(
    name = "main_kernel",
    srcs = ["main_kernel.cc"],
    deps = [":kernel"],
)
