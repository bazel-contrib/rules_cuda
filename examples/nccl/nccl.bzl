load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")

# NOTE: all paths in this file relative to @nccl repo root.

def if_cuda_nvcc(if_true, if_false = []):
    return select({
        "@rules_cuda//cuda:compiler_is_nvcc": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_clang(if_true, if_false = []):
    return select({
        "@rules_cuda//cuda:compiler_is_clang": if_true,
        "//conditions:default": if_false,
    })

def nccl_primitive(name, hdrs = [], deps = [], use_bf16 = True):
    ops = ["sum", "prod", "min", "max", "premulsum", "sumpostdiv"]
    datatypes = ["i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", "f64"]
    if use_bf16:
        datatypes.append("bf16")

    intermediate_targets = []
    for opn, op in enumerate(ops):
        for dtn, dt in enumerate(datatypes):
            name_op_dt = "{}_{}_{}".format(name, op, dt)
            copy_file(
                name = name_op_dt + "_rename",
                src = "src/collectives/device/{}.cu".format(name),
                out = "src/collectives/device/{}.cu".format(name_op_dt),
            )

            cuda_objects(
                name = name_op_dt,
                srcs = [":{}_rename".format(name_op_dt)],
                hdrs = hdrs,
                deps = deps,
                ptxasopts = ["-maxrregcount=96"],
                defines = ["NCCL_OP={}".format(opn), "NCCL_TYPE={}".format(dtn)],
                includes = ["src/collectives/device"],
            )
            intermediate_targets.append(":" + name_op_dt)

    cuda_objects(name = name, deps = intermediate_targets)
