load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaObjectsInfo")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain")
load("//cuda/private:actions/nvcc_compile.bzl", "compile")

def _cuda_objects_impl(ctx):
    allow_hdrs = [".cuh", ".h", ".hpp", "hh"]
    allow_srcs = [".cu", ".cu.cc"]
    cuda_helper.check_srcs_extensions(ctx, allow_hdrs + allow_srcs , "cuda_object")

    # outputs
    objects = []
    rdc_objects = []
    pic_objects = []
    rdc_pic_objects = []

    cc_toolchain = find_cpp_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)

    includes = depset()
    system_includes = depset()
    quote_includes = depset()
    private_headers = []
    for src in ctx.attr.srcs:
        hdrs = [f for f in src.files.to_list() if cuda_helper.check_src_extension(f, allow_hdrs)]
        private_headers.append(depset(hdrs))
    headers = depset(transitive = private_headers + [hdr[DefaultInfo].files for hdr in ctx.attr.hdrs])

    for src in ctx.attr.srcs:
        files = src[DefaultInfo].files.to_list()
        for translation_unit in files:
            basename = cuda_helper.get_basename_without_ext(translation_unit.basename, allow_srcs, fail_if_not_match=False)
            if not basename:
                continue
            objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, pic = False, rdc = False))
            rdc_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, pic = False, rdc = True))
            pic_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, pic = True, rdc = False))
            rdc_pic_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, pic = True, rdc = True))

    objects = depset(objects)
    pic_objects = depset(pic_objects)
    rdc_objects = depset(rdc_objects)
    rdc_pic_objects = depset(rdc_pic_objects)

    return [
        DefaultInfo(
            files = depset(transitive=[objects, pic_objects, rdc_objects, rdc_pic_objects]),
        ),
        OutputGroupInfo(
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            rdc_pic_objects = rdc_pic_objects,
        ),
        CudaObjectsInfo(
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            rdc_pic_objects = rdc_pic_objects,
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
