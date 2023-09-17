load("@rules_cc//cc:defs.bzl", "cc_test")
load("//cuda/private:rules/cuda_library.bzl", "cuda_library")

def cuda_test(
        name,
        srcs = [],
        copts = [],
        visibility = ["//visibility:private"],
        data = [],
        deps = []):
    cuda_library_name = "_" + name

    cuda_library(
        name = cuda_library_name,
        srcs = srcs,
        copts = copts,
        deps = deps,
    )

    cc_test(
        name = name,
        copts = copts,
        visibility = visibility,
        data = data,
        deps = [cuda_library_name],
    )
