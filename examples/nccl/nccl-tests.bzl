load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# NOTE: all paths in this file relative to @nccl-tests repo root.

def nccl_tests_binary(name, cc_deps = [], cuda_deps = []):
    cuda_library(
        name = name,
        srcs = ["src/{}.cu".format(name)],
        deps = [
            "@nccl//:nccl_shared",
            ":common_cuda",
        ],
        alwayslink = 1,
    )

    bin_name = name + "_perf"
    native.cc_binary(
        name = bin_name,
        deps = [":common_cc", ":" + name],
        visibility = ["//visibility:public"],
    )
