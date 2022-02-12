load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaObjectsInfo")
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

    # inputs
    objects = []
    pic_objects = []

    for src in attr.srcs:
        files = src[DefaultInfo].files.to_list()
        for translation_unit in files:
            basename = cuda_helper.get_basename_without_ext(translation_unit.basename, ALLOW_CUDA_SRCS, fail_if_not_match = False)
            if not basename:
                continue
            objects.append(compile(ctx, cuda_toolchain, cc_toolchain, translation_unit, basename, includes = common.includes, system_includes = common.system_includes, quote_includes = common.quote_includes, headers = common.headers, defines = common.defines, compile_flags = common.compile_flags, host_defines = common.host_defines + common.host_local_defines, host_compile_flags = common.host_compile_flags, pic = False, rdc = attr.rdc))
            pic_objects.append(compile(ctx, cuda_toolchain, cc_toolchain, translation_unit, basename, includes = common.includes, system_includes = common.system_includes, quote_includes = common.quote_includes, headers = common.headers, defines = common.defines, compile_flags = common.compile_flags, host_defines = common.host_defines + common.host_local_defines, host_compile_flags = common.host_compile_flags, pic = True, rdc = attr.rdc))

    objects = depset(objects)
    pic_objects = depset(pic_objects)

    compilation_ctx = cc_common.create_compilation_context(
        headers = common.headers,
        includes = common.includes,
        system_includes = common.system_includes,
        quote_includes = common.quote_includes,
        defines = depset(common.host_defines),
        local_defines = depset(common.host_local_defines),
    )

    if attr.rdc:
        transitive_objects = depset(transitive = [dep[CudaObjectsInfo].rdc_objects for dep in attr.deps if CudaObjectsInfo in dep])
        transitive_pic_objects = depset(transitive = [dep[CudaObjectsInfo].rdc_pic_objects for dep in attr.deps if CudaObjectsInfo in dep])
        objects = depset(transitive = [objects, transitive_objects])
        pic_objects = depset(transitive = [pic_objects, transitive_pic_objects])
        dlink_object = depset([device_link(ctx, cuda_toolchain, cc_toolchain, objects, attr.name, link_flags = common.link_flags, pic = False, rdc = attr.rdc)])
        dlink_pic_object = depset([device_link(ctx, cuda_toolchain, cc_toolchain, pic_objects, attr.name, link_flags = common.link_flags, pic = True, rdc = attr.rdc)])
        objects = depset(transitive = [objects, dlink_object])
        pic_objects = depset(transitive = [pic_objects, dlink_pic_object])

    lib = create_library(ctx, cuda_toolchain, objects, attr.name, pic = False)
    pic_lib = create_library(ctx, cuda_toolchain, pic_objects, attr.name, pic = True)

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
            cc_common.create_linker_input(owner = ctx.label, libraries = depset([lib_to_link])),
        ], transitive = common.transitive_linker_inputs),
    )

    return [
        DefaultInfo(
            files = depset([lib, pic_lib]),
        ),
        OutputGroupInfo(
            lib = [lib],
            pic_lib = [pic_lib],
        ),
        CcInfo(
            compilation_context = compilation_ctx,
            linking_context = linking_ctx,
        ),
    ]

cuda_library = rule(
    implementation = _cuda_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS),
        "hdrs": attr.label_list(allow_files = ALLOW_CUDA_HDRS),
        "deps": attr.label_list(providers = [[CcInfo], [CudaObjectsInfo]]),
        "alwayslink": attr.bool(default = False),
        "rdc": attr.bool(default = False, doc = "whether to perform relocateable device code linking, otherwise, normal device link."),
        "includes": attr.string_list(doc = "List of include dirs to be added to the compile line."),
        # host_* attrs will be passed transitively to cc_* and cuda_* targets
        "host_copts": attr.string_list(doc = "Add these options to the CUDA host compilation command."),
        "host_defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "host_local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        "host_linkopts": attr.string_list(doc = "Add these flags to the host linker command."),
        # non-host attrs will be passed transitively to cuda_* targets only.
        "copts": attr.string_list(doc = "Add these options to the CUDA device compilation command."),
        "defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        "linkopts": attr.string_list(doc = "Add these flags to the CUDA linker command."),
        "_builtin_deps": attr.label_list(default = ["@rules_cuda//cuda:runtime"]),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
        "_default_cuda_archs": attr.label(default = "@rules_cuda//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = ["//cuda:toolchain_type"],
)
