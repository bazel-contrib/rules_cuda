load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaObjectsInfo")


def compile(ctx, translation_unit, headers, includes, system_includes, quote_includes, , pic = False, rdc = False)


def _cuda_objects_impl(ctx):
    allow_exts = [".cu", ".cu.cc"]
    cuda_helper.check_srcs_extensions(ctx, allow_exts, "cuda_object")

    cc_toolchain = find_cpp_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)

    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    # TODO: platform dependent
    obj_ext = ".o"
    rdc_ext = ".rdc"
    pic_ext = ".pic"

    args = ctx.actions.args()
    args.add("-ccbin", host_compiler)
    # args.add("-Xcompiler", "-###")
    args.add("-x", "cu")
    # args.add("-objtemp")

    objects = []
    rdc_objects = []
    pic_objects = []
    pic_rdc_objects = []

    inputs = depset(transitive = [hdr[DefaultInfo].files for hdr in ctx.attr.hdrs if DefaultInfo in hdr ])
    for src in ctx.attr.srcs:
        if DefaultInfo in src:
            files = src[DefaultInfo].files.to_list()
            for file in files:
                name = cuda_helper.get_basename_without_ext(file.basename, allow_exts)
                obj_file = ctx.actions.declare_file(name + obj_ext)
                ctx.actions.run(
                    executable = cuda_compiler,
                    arguments = [args, "--device-c", file.path, "-o", obj_file.path],
                    outputs = [obj_file],
                    inputs = depset([file], transitive = [inputs, cc_toolchain.all_files]),
                    env = {
                        "PATH": "/usr/bin",
                        # "TMPDIR": "/tmmmmmm",
                    }
                )
                objects.append(obj_file)

    return [
        DefaultInfo(
            files = depset(objects + pic_objects + rdc_objects + pic_rdc_objects)
        ),
        OutputGroupInfo(
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            pic_rdc_objects = pic_rdc_objects,
        ),
        CudaObjectsInfo(
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            pic_rdc_objects = pic_rdc_objects,
        ),
    ]

cuda_objects = rule(
    implementation = _cuda_objects_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cu", ".cuh", ".cu.cc"]),
        "hdrs": attr.label_list(allow_files = [".cuh", ".h", ".hpp", "hh"]),
        "deps": attr.label_list(providers = [[CcInfo], [CudaObjectsInfo]]),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    fragments = ["cpp"],
    toolchains = ["//cuda:toolchain_type"],
)
