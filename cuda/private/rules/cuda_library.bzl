load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:actions/compile.bzl", "compile")
load("//cuda/private:actions/dlink.bzl", "device_link")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaInfo")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS", "ALLOW_CUDA_SRCS")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain", "use_cpp_toolchain", "use_cuda_toolchain")

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

    # flatten first, so that non-unique basenames can be properly deduplicated
    src_files = []
    for src in ctx.attr.srcs:
        src_files.extend(src[DefaultInfo].files.to_list())

    # merge deps' direct objects and archive objects as our archive objects
    archive_objects = depset(transitive = [dep[CudaInfo].objects for dep in attr.deps if CudaInfo in dep] +
                                          [dep[CudaInfo].archive_objects for dep in attr.deps if CudaInfo in dep])
    archive_pic_objects = depset(transitive = [dep[CudaInfo].pic_objects for dep in attr.deps if CudaInfo in dep] +
                                              [dep[CudaInfo].archive_pic_objects for dep in attr.deps if CudaInfo in dep])
    archive_rdc_objects = depset(transitive = [dep[CudaInfo].rdc_objects for dep in attr.deps if CudaInfo in dep] +
                                              [dep[CudaInfo].archive_rdc_objects for dep in attr.deps if CudaInfo in dep])
    archive_rdc_pic_objects = depset(transitive = [dep[CudaInfo].rdc_pic_objects for dep in attr.deps if CudaInfo in dep] +
                                                  [dep[CudaInfo].archive_rdc_pic_objects for dep in attr.deps if CudaInfo in dep])

    # Gather transitive dlink objects that may come from other `cuda_library`s
    dlink_rdc_objects = depset(transitive = [dep[CudaInfo].dlink_rdc_objects for dep in attr.deps if CudaInfo in dep])
    dlink_rdc_pic_objects = depset(transitive = [dep[CudaInfo].dlink_rdc_pic_objects for dep in attr.deps if CudaInfo in dep])

    # direct outputs
    objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = False, rdc = False)) if not use_rdc else depset([])
    pic_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = True, rdc = False)) if not use_rdc else depset([])
    rdc_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = False, rdc = True)) if use_rdc else depset([])
    rdc_pic_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = True, rdc = True)) if use_rdc else depset([])

    # if rdc is enabled for this `cuda_library`, then we need to do a pass of device link further.
    rdc_dlink_inputs = None
    rdc_pic_dlink_inputs = None
    if use_rdc:
        # TODO: Switch to explicit dlink with attr `dlink=True`, then add support dlink with libraries. At the moment,
        # all libraries produced by this rule with `rdc=True` will have an <name>_dlink.<infix>.o archived, and nvlink
        # refuses to consume such libraries and ignores them silently.

        # prepare inputs for device_link, take use_rdc=True and non-pic as an example:
        # rdc_objects: produce with this rule
        # archive_rdc_objects: propagate from other `cuda_objects`
        # dlink_rdc_objects: propagate from other `cuda_library`s
        rdc_dlink_inputs = depset(transitive = [rdc_objects, archive_rdc_objects, dlink_rdc_objects])
        rdc_pic_dlink_inputs = depset(transitive = [rdc_pic_objects, archive_rdc_pic_objects, dlink_rdc_pic_objects])

        rdc_dlink_output = depset([device_link(ctx, cuda_toolchain, cc_toolchain, rdc_dlink_inputs, common, pic = False, rdc = True)])
        rdc_pic_dlink_output = depset([device_link(ctx, cuda_toolchain, cc_toolchain, rdc_pic_dlink_inputs, common, pic = True, rdc = True)])

        # update the **direct** outputs
        rdc_objects = depset(transitive = [rdc_objects, rdc_dlink_output])
        rdc_pic_objects = depset(transitive = [rdc_pic_objects, rdc_pic_dlink_output])

    # objects to archive: objects directly outputted by this rule and all objects transitively from deps,
    # take use_rdc=True and non-pic as an example:
    # rdc_objects: produce with this rule, thus it must be archived in the library produced by this rule
    # archive_rdc_objects: propagate from other `cuda_objects`, so this rule is in charge of archiving them
    # dlink_rdc_objects is NOT included!
    if not use_rdc:
        archive_content = depset(transitive = [objects, archive_objects])
        pic_archive_content = depset(transitive = [pic_objects, archive_pic_objects])
    else:
        archive_content = depset(transitive = [rdc_objects, archive_rdc_objects])
        pic_archive_content = depset(transitive = [rdc_pic_objects, archive_rdc_pic_objects])

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
        compilation_outputs = cc_common.create_compilation_outputs(objects = archive_content, pic_objects = pic_archive_content),
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

    cc_info = cc_common.merge_cc_infos(
        direct_cc_infos = [CcInfo(compilation_context = compilation_ctx, linking_context = linking_ctx)],
        cc_infos = [common.transitive_cc_info],
    )

    return [
        DefaultInfo(files = depset(libs + pic_libs)),
        OutputGroupInfo(
            lib = libs,
            pic_lib = pic_libs,
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            rdc_pic_objects = rdc_pic_objects,
        ),
        CcInfo(
            compilation_context = cc_info.compilation_context,
            linking_context = cc_info.linking_context,
        ),
        cuda_helper.create_cuda_info(
            defines = depset(common.defines),
            # all objects from cuda_objects should be properly archived, thus, the transitivity of archive is cut off here.
            dlink_rdc_objects = rdc_dlink_inputs,
            dlink_rdc_pic_objects = rdc_pic_dlink_inputs,
        ),
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
        "rdc": attr.bool(
            default = False,
            doc = ("Whether to perform device linking for relocatable device code. " +
                   "Transitive deps that contain device code must all either be cuda_objects or cuda_library(rdc = True)."),
        ),
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
        "_default_cuda_copts": attr.label(default = "//cuda:copts"),
        "_default_host_copts": attr.label(default = "//cuda:host_copts"),
        "_default_cuda_archs": attr.label(default = "//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain() + use_cuda_toolchain(),
    provides = [DefaultInfo, OutputGroupInfo, CcInfo, CudaInfo],
)
