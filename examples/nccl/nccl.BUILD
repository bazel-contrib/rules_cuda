load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")
load("@rules_cuda_examples//nccl:nccl.bzl", "if_cuda_clang", "if_cuda_nvcc", "nccl_primitive")

# NOTE: all paths in this file relative to @nccl repo root.

expand_template(
    name = "nccl_h",
    out = "src/include/nccl.h",
    substitutions = {
        "${nccl:Major}": "2",
        "${nccl:Minor}": "18",
        "${nccl:Patch}": "3",
        "${nccl:Suffix}": "",
        # NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
        "${nccl:Version}": "21803",
    },
    template = "src/nccl.h.in",
)

cc_library(
    name = "nccl_include",
    hdrs = [
        ":nccl_h",
    ] + glob([
        "src/include/**/*.h",
        "src/include/**/*.hpp",
    ]),
    includes = [
        # this will add both nccl/src/include in repo and
        # bazel-out/<compilation_mode>/bin/nccl/src/include to include paths
        # so the previous expand_template generate nccl.h to the very path!
        "src/include",
    ],
)

cuda_objects(
    name = "nccl_device_common",
    srcs = [
        "src/collectives/device/functions.cu",
        "src/collectives/device/onerank_reduce.cu",
    ] + glob([
        "src/collectives/device/**/*.h",
    ]),
    copts = if_cuda_nvcc(["--extended-lambda"]),
    ptxasopts = ["-maxrregcount=96"],
    deps = [":nccl_include"],
)

# must be manually disabled if cuda version is lower than 11.
USE_BF16 = True

filegroup(
    name = "collective_dev_hdrs",
    srcs = [
        "src/collectives/device/all_gather.h",
        "src/collectives/device/all_reduce.h",
        "src/collectives/device/broadcast.h",
        "src/collectives/device/common.h",
        "src/collectives/device/common_kernel.h",
        "src/collectives/device/gen_rules.sh",
        "src/collectives/device/op128.h",
        "src/collectives/device/primitives.h",
        "src/collectives/device/prims_ll.h",
        "src/collectives/device/prims_ll128.h",
        "src/collectives/device/prims_simple.h",
        "src/collectives/device/reduce.h",
        "src/collectives/device/reduce_kernel.h",
        "src/collectives/device/reduce_scatter.h",
        "src/collectives/device/sendrecv.h",
    ],
)

# cuda_objects for each type of primitive
nccl_primitive(
    name = "all_gather",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

nccl_primitive(
    name = "all_reduce",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

nccl_primitive(
    name = "broadcast",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

nccl_primitive(
    name = "reduce",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

nccl_primitive(
    name = "reduce_scatter",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

nccl_primitive(
    name = "sendrecv",
    hdrs = ["collective_dev_hdrs"],
    use_bf16 = USE_BF16,
    deps = [":nccl_device_common"],
)

# device link
cuda_library(
    name = "collectives",
    rdc = 1,
    deps = [
        ":all_gather",
        ":all_reduce",
        ":broadcast",
        ":reduce",
        ":reduce_scatter",
        ":sendrecv",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "nccl",
    srcs = glob(
        [
            "src/*.cc",
            "src/collectives/*.cc",
            "src/graph/*.cc",
            "src/graph/*.h",
            "src/misc/*.cc",
            "src/transport/*.cc",
        ],
        exclude = [
            # https://github.com/NVIDIA/nccl/issues/658
            "src/enhcompat.cc",
        ],
    ),
    copts = if_cuda_clang(["-xcu"]),
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":collectives",
        ":nccl_include",
        "@rules_cuda//cuda:runtime",
    ],
)

# To allow downstream targets to link with the nccl shared library, we need to `cc_import` it again.
# See https://groups.google.com/g/bazel-discuss/c/RtbidPdVFyU/m/TsUDOVHIAwAJ
cc_import(
    name = "nccl_shared",
    shared_library = ":nccl",
    visibility = ["//visibility:public"],
)
