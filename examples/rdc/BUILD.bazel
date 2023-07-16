load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")

cuda_objects(
    name = "a_objects",
    srcs = ["a.cu"],
    deps = [":b_objects"],
)

cuda_objects(
    name = "b_objects",
    srcs = ["b.cu"],
    hdrs = ["b.cuh"],
)

cuda_library(
    name = "lib_from_objects",
    rdc = True,
    deps = [
        ":a_objects",
        ":b_objects",
    ],
)

cc_binary(
    name = "main_from_objects",
    deps = [
        ":lib_from_objects",
    ],
)

# Another way of doing it is to just use cuda_library
cuda_library(
    name = "a",
    srcs = ["a.cu"],
    rdc = True,
    deps = [":b"],
)

cuda_library(
    name = "b",
    srcs = ["b.cu"],
    hdrs = ["b.cuh"],
    rdc = True,
)

cc_binary(
    name = "main_from_library",
    deps = [
        ":a",
        ":b",
    ],
)
