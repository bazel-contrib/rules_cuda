load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaArchsInfo", "CudaInfo")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain", "use_cpp_toolchain", "use_cuda_toolchain")
load("//cuda/private:actions/nvcc_compile.bzl", "compile")
load("//cuda/private:actions/nvcc_dlink.bzl", "device_link")
load("//cuda/private:actions/nvcc_lib.bzl", "create_library")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS", "ALLOW_CUDA_SRCS")

def _cuda_library_impl(ctx):
    """cuda_library is a rule that perform device link.

    cuda_library produce self-contained object file. It produces object files
    or static library that is consumable by cc_* rules"""

    attr = ctx.attr
    cuda_helper.check_srcs_extensions(ctx, ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS, "cuda_library")

    cc_toolchain = find_cpp_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)

    common = cuda_helper.create_common(ctx)
    use_rdc = attr.rdc
    if not use_rdc:
        use_rdc = cuda_helper.check_must_enforce_rdc(cuda_archs_info = ctx.attr._default_cuda_archs[CudaArchsInfo])

    # outputs
    objects = []
    pic_objects = []

    for src in attr.srcs:
        files = src[DefaultInfo].files.to_list()
        objects.extend(compile(ctx, cuda_toolchain, cc_toolchain, files, common, pic = False, rdc = use_rdc))
        pic_objects.extend(compile(ctx, cuda_toolchain, cc_toolchain, files, common, pic = True, rdc = use_rdc))

    objects = depset(objects)
    pic_objects = depset(pic_objects)

    # if rdc is enabled for this cuda_library, then we need futher do a pass of device link
    if use_rdc:
        transitive_objects = depset(transitive = [dep[CudaInfo].rdc_objects for dep in attr.deps if CudaInfo in dep])
        transitive_pic_objects = depset(transitive = [dep[CudaInfo].rdc_pic_objects for dep in attr.deps if CudaInfo in dep])
        objects = depset(transitive = [objects, transitive_objects])
        pic_objects = depset(transitive = [pic_objects, transitive_pic_objects])
        dlink_object = depset([device_link(ctx, cuda_toolchain, cc_toolchain, objects, common, pic = False, rdc = use_rdc)])
        dlink_pic_object = depset([device_link(ctx, cuda_toolchain, cc_toolchain, pic_objects, common, pic = True, rdc = use_rdc)])
        objects = depset(transitive = [objects, dlink_object])
        pic_objects = depset(transitive = [pic_objects, dlink_pic_object])

    compilation_ctx = cc_common.create_compilation_context(
        headers = common.headers,
        includes = depset(common.includes),
        quote_includes = depset(common.quote_includes),
        system_includes = depset(common.system_includes),
        defines = depset(common.host_defines),
        local_defines = depset(common.host_local_defines),
    )

    lib = create_library(ctx, cuda_toolchain, cc_toolchain, objects, pic = False)
    pic_lib = create_library(ctx, cuda_toolchain, cc_toolchain, pic_objects, pic = True)

    lib_to_link = cc_common.create_library_to_link(
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        static_library = lib,
        pic_static_library = pic_lib,
        alwayslink = attr.alwayslink,
        # pic_objects = pic_objects, // Experimental, do not use
        # objects = objects, // Experimental, do not use
    )
    linking_ctx = cc_common.create_linking_context(
        linker_inputs = depset([
            cc_common.create_linker_input(owner = ctx.label, libraries = depset([lib_to_link]), user_link_flags = common.host_link_flags),
        ], transitive = common.transitive_linker_inputs),
    )

    return [
        DefaultInfo(
            files = depset([
                lib,
                # pic_lib,
            ]),
        ),
        OutputGroupInfo(
            lib = [lib],
            pic_lib = [pic_lib],
        ),
        CcInfo(
            compilation_context = compilation_ctx,
            linking_context = linking_ctx,
        ),
        cuda_helper.create_cuda_info(defines = depset(common.defines)),
    ]

cuda_library = rule(
    implementation = _cuda_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS),
        "hdrs": attr.label_list(allow_files = ALLOW_CUDA_HDRS),
        "deps": attr.label_list(providers = [[CcInfo], [CudaInfo]]),
        "alwayslink": attr.bool(default = False),
        "rdc": attr.bool(default = False, doc = "whether to perform relocateable device code linking, otherwise, normal device link."),
        "includes": attr.string_list(doc = "List of include dirs to be added to the compile line."),
        "host_copts": attr.string_list(doc = "Add these options to the CUDA host compilation command."),
        "host_defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "host_local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        "host_linkopts": attr.string_list(doc = "Add these flags to the host library link command."),
        "copts": attr.string_list(doc = "Add these options to the CUDA device compilation command."),
        "defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        "linkopts": attr.string_list(doc = "Add these flags to the CUDA device link command."),
        "_builtin_deps": attr.label_list(default = ["@rules_cuda//cuda:runtime"]),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  # legacy behaviour
        "_default_cuda_archs": attr.label(default = "@rules_cuda//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain() + use_cuda_toolchain(),
)
