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

# Run with 'bazel run --cuda //examples:main_cublas'
cc_binary(
    name = "main_cublas",
    srcs = ["main_cublas.cc"],
    deps = ["@local_cuda//:cublas"],
)

# thrust have code marked as __global__ or __kernel__
# so source for thrust should be built with cuda_library first
cuda_library(
    name = "main_thrust_cu",
    srcs = ["main_thrust.cc"],
    deps = ["@local_cuda//:thrust"],
)

# Run with 'bazel run --cuda //examples:main_thrust'
cc_binary(
    name = "main_thrust",
    deps = [":main_thrust_cu"],
)
