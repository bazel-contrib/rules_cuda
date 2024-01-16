load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cuda_examples//nccl:nccl-tests.bzl", "nccl_tests_binary")

# NOTE: all paths in this file relative to @nccl-tests repo root.

cc_library(
    name = "nccl_tests_include",
    hdrs = glob(["src/*.h"]),
    includes = ["src"],
)

cuda_library(
    name = "common_cuda",
    srcs = [
        "src/common.cu",
        "verifiable/verifiable.cu",
    ] + glob([
        "**/*.h",
    ]),
    deps = [
        ":nccl_tests_include",
        "@nccl",
    ],
)

cc_library(
    name = "common_cc",
    srcs = ["src/timer.cc"],
    hdrs = ["src/timer.h"],
    alwayslink = 1,
)

# :common_cuda, :common_cc and @nccl//:nccl_shared are implicitly hardcoded in `nccl_tests_binary`
nccl_tests_binary(name = "all_reduce")

nccl_tests_binary(name = "all_gather")

nccl_tests_binary(name = "broadcast")

nccl_tests_binary(name = "reduce_scatter")

nccl_tests_binary(name = "reduce")

nccl_tests_binary(name = "alltoall")

nccl_tests_binary(name = "scatter")

nccl_tests_binary(name = "gather")

nccl_tests_binary(name = "sendrecv")

nccl_tests_binary(name = "hypercube")
