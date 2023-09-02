load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")
load("@rules_cuda_examples//nccl:nccl.bzl", "if_cuda_clang", "if_cuda_nvcc", "nccl_primitive")

expand_template(
    name = "nccl_h",
    out = "nccl/src/include/nccl.h",
    substitutions = {
        "${nccl:Major}": "2",
        "${nccl:Minor}": "18",
        "${nccl:Patch}": "3",
        "${nccl:Suffix}": "",
        # NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
        "${nccl:Version}": "21803",
    },
    template = "nccl/src/nccl.h.in",
)

cc_library(
    name = "nccl_include",
    hdrs = [
        ":nccl_h",
    ] + glob([
        "nccl/src/include/**/*.h",
        "nccl/src/include/**/*.hpp",
    ]),
    includes = [
        # this will add both nccl/src/include in repo and
        # bazel-out/<compilation_mode>/bin/nccl/src/include to include paths
        # so the previous expand_template generate nccl.h to the very path!
        "nccl/src/include",
    ],
)

cuda_objects(
    name = "nccl_device_common",
    srcs = [
        "nccl/src/collectives/device/functions.cu",
        "nccl/src/collectives/device/onerank_reduce.cu",
    ] + glob([
        "nccl/src/collectives/device/**/*.h",
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
        "nccl/src/collectives/device/all_gather.h",
        "nccl/src/collectives/device/all_reduce.h",
        "nccl/src/collectives/device/broadcast.h",
        "nccl/src/collectives/device/common.h",
        "nccl/src/collectives/device/common_kernel.h",
        "nccl/src/collectives/device/gen_rules.sh",
        "nccl/src/collectives/device/op128.h",
        "nccl/src/collectives/device/primitives.h",
        "nccl/src/collectives/device/prims_ll.h",
        "nccl/src/collectives/device/prims_ll128.h",
        "nccl/src/collectives/device/prims_simple.h",
        "nccl/src/collectives/device/reduce.h",
        "nccl/src/collectives/device/reduce_kernel.h",
        "nccl/src/collectives/device/reduce_scatter.h",
        "nccl/src/collectives/device/sendrecv.h",
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
            "nccl/src/*.cc",
            "nccl/src/collectives/*.cc",
            "nccl/src/graph/*.cc",
            "nccl/src/graph/*.h",
            "nccl/src/misc/*.cc",
            "nccl/src/transport/*.cc",
        ],
        exclude = [
            # https://github.com/NVIDIA/nccl/issues/658
            "nccl/src/enhcompat.cc",
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
