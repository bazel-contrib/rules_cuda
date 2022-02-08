load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaObjectsInfo", "CudaArchsInfo")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain")
load("//cuda/private:actions/nvcc_compile.bzl", "compile")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS", "ALLOW_CUDA_SRCS")

def _cuda_objects_impl(ctx):
    cuda_helper.check_srcs_extensions(ctx, ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS , "cuda_object")

    # outputs
    objects = []
    rdc_objects = []
    pic_objects = []
    rdc_pic_objects = []

    cc_toolchain = find_cpp_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)

    compile_arch_flags = cuda_helper.get_nvcc_compile_arch_flags(ctx.attr._default_cuda_archs[CudaArchsInfo].arch_specs)

    includes = depset()
    system_includes = depset()
    quote_includes = depset()
    private_headers = []
    for src in ctx.attr.srcs:
        hdrs = [f for f in src.files.to_list() if cuda_helper.check_src_extension(f, ALLOW_CUDA_HDRS)]
        private_headers.append(depset(hdrs))
    headers = depset(transitive = private_headers + [hdr[DefaultInfo].files for hdr in ctx.attr.hdrs])

    for src in ctx.attr.srcs:
        files = src[DefaultInfo].files.to_list()
        for translation_unit in files:
            basename = cuda_helper.get_basename_without_ext(translation_unit.basename, ALLOW_CUDA_SRCS, fail_if_not_match=False)
            if not basename:
                continue
            objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, compile_arch_flags, pic = False, rdc = False))
            rdc_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, compile_arch_flags, pic = False, rdc = True))
            pic_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, compile_arch_flags, pic = True, rdc = False))
            rdc_pic_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, basename, compile_arch_flags, pic = True, rdc = True))

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
        "srcs": attr.label_list(allow_files = ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS),
        "hdrs": attr.label_list(allow_files = ALLOW_CUDA_HDRS),
        "deps": attr.label_list(providers = [[CcInfo], [CudaObjectsInfo]]),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
        "_default_cuda_archs": attr.label(default = "@rules_cuda//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = ["//cuda:toolchain_type"],
)
