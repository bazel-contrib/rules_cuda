load("@rules_cuda//cuda:defs.bzl", "cuda_binary", "cuda_test")

package(default_visibility = ["//visibility:public"])

cuda_binary(
    name = "main",
    srcs = ["main.cu"],
)

cuda_test(
    name = "test",
    srcs = ["main.cu"],
)
