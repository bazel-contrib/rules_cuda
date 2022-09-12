load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaInfo")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain", "use_cpp_toolchain", "use_cuda_toolchain")
load("//cuda/private:actions/compile.bzl", "compile")
load("//cuda/private:actions/dlink_wrapper.bzl", "device_link")
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
        use_rdc = cuda_helper.check_must_enforce_rdc(cuda_archs_info = common.cuda_archs_info)

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

    cc_feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    linking_ctx, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.attr.name,
        actions = ctx.actions,
        feature_configuration = cc_feature_config,
        cc_toolchain = cc_toolchain,
        compilation_outputs = cc_common.create_compilation_outputs(objects = objects, pic_objects = pic_objects),
        user_link_flags = common.host_link_flags,
        alwayslink = attr.alwayslink,
        linking_contexts = common.transitive_linking_contexts,
        disallow_dynamic_library = True,
    )

    lib = None
    pic_lib = None
    if linking_outputs.library_to_link != None:
        lib = linking_outputs.library_to_link.static_library
        pic_lib = linking_outputs.library_to_link.pic_static_library
    libs = [] if lib == None else [lib]
    pic_libs = [] if pic_lib == None else [pic_lib]

    return [
        DefaultInfo(files = depset(libs + pic_libs)),
        OutputGroupInfo(
            lib = libs,
            pic_lib = pic_libs,
            objects = objects,
            pic_objects = pic_objects,
        ),
        CcInfo(
            compilation_context = compilation_ctx,
            linking_context = linking_ctx,
        ),
        cuda_helper.create_cuda_info(defines = depset(common.defines)),
    ]

cuda_library = rule(
    doc = """This rule compiles and creates static library for CUDA kernel code. The resulting targets can then be consumed by
[C/C++ Rules](https://bazel.build/reference/be/c-cpp#rules).""",
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
        "ptxasopts": attr.string_list(doc = "Add these flags to the ptxas command."),
        "_builtin_deps": attr.label_list(default = ["//cuda:runtime"]),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  # legacy behaviour
        "_default_host_copts": attr.label(default = "//cuda:copts"),
        "_default_cuda_archs": attr.label(default = "//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain() + use_cuda_toolchain(),
)
