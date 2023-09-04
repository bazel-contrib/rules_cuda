load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cuda_examples//nccl:nccl-tests.bzl", "nccl_tests_binary")

cc_library(
    name = "nccl_tests_include",
    hdrs = glob(["nccl-tests/src/*.h"]),
    includes = ["nccl-tests/src"],
)

cuda_library(
    name = "common_cuda",
    srcs = [
        "nccl-tests/src/common.cu",
        "nccl-tests/verifiable/verifiable.cu",
    ] + glob([
        "nccl-tests/**/*.h",
    ]),
    deps = [
        ":nccl_tests_include",
        "@nccl",
    ],
)

cc_library(
    name = "common_cc",
    srcs = ["nccl-tests/src/timer.cc"],
    hdrs = ["nccl-tests/src/timer.h"],
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
