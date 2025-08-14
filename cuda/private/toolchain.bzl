load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper")

def _get_cc_sysroot(ctx, cc_toolchain):
    """Extract sysroot from cc toolchain"""
    if getattr(cc_toolchain, "sysroot", None):
        return cc_toolchain.sysroot
    else:
        # Fallback to extracting sysroot from compiler flags
        feature_configuration = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        )

        variables = cc_common.create_compile_variables(
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.cxxopts,
        )

        cc_flags = cc_common.get_memory_inefficient_command_line(
            feature_configuration = feature_configuration,
            action_name = CC_ACTION_NAMES.cpp_compile,
            variables = variables,
        )

        for flag in cc_flags:
            if flag.startswith("--sysroot="):
                return flag.removeprefix("--sysroot=")

    return None

def _cuda_toolchain_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    has_compiler_executable = ctx.attr.compiler_executable != None and ctx.attr.compiler_executable != ""
    has_compiler_label = ctx.attr.compiler_label != None
    compiler_executable = None

    if cc_toolchain == None:
        fail("cuda toolchain needs a valid cc toolchain to be useful")

    # Validation
    # compiler_use_cc_toolchain should be used alone and not along with compiler_executable or compiler_label
    if (ctx.attr.compiler_use_cc_toolchain == True) and (has_compiler_executable or has_compiler_label):
        fail("compiler_use_cc_toolchain set to True but compiler_executable or compiler_label also set.")
    elif (ctx.attr.compiler_use_cc_toolchain == False) and not has_compiler_executable and not has_compiler_label:
        fail("Either compiler_executable or compiler_label must be specified or if a valid cc_toolchain is registered, set attr compiler_use_cc_toolchain to True.")

    # First, attempt to use configured cc_toolchain if attr compiler_use_cc_toolchain set.
    if (ctx.attr.compiler_use_cc_toolchain == True):
        compiler_executable = cc_toolchain.compiler_executable
    elif has_compiler_executable:
        compiler_executable = ctx.attr.compiler_executable
    elif has_compiler_label:
        l = ctx.attr.compiler_label.label
        compiler_executable = "{}/{}/{}".format(l.workspace_root, l.package, l.name)

    cuda_toolchain_config = ctx.attr.toolchain_config[CudaToolchainConfigInfo]
    selectables_info = config_helper.collect_selectables_info(cuda_toolchain_config.action_configs + cuda_toolchain_config.features)
    must_have_selectables = []
    for name in must_have_selectables:
        if not config_helper.is_configured(selectables_info, name):
            fail(name, "is not configured (not exists) in the provided toolchain_config")

    artifact_name_patterns = {}
    for pattern in cuda_toolchain_config.artifact_name_patterns:
        artifact_name_patterns[pattern.category_name] = pattern

    # construct compiler_depset
    compiler_depset = depset()
    if ctx.attr.compiler_use_cc_toolchain:
        compiler_depset = cc_toolchain.all_files  # pass all cc_toolchain to toolchain_files
    elif has_compiler_executable:
        pass
    elif has_compiler_label:
        compiler_target_info = ctx.attr.compiler_label[DefaultInfo]
        if not compiler_target_info.files_to_run or not compiler_target_info.files_to_run.executable:
            fail("compiler_label specified is not an executable, specify a valid compiler_label")
        compiler_depset = depset(direct = [compiler_target_info.files_to_run.executable], transitive = [compiler_target_info.default_runfiles.files])

    toolchain_files = depset(transitive = [compiler_depset] + [cf.files for cf in ctx.attr.compiler_files])

    optional_attributes = {}
    cc_sysroot = _get_cc_sysroot(ctx, cc_toolchain)
    if cc_sysroot != None:
        # NOTE: cuda_toolchain may or may not be configured with cc_sysroot.
        # If cuda_toolchain has the attr, it will be correct. Otherwise, it is not configured.
        optional_attributes["cc_sysroot"] = cc_sysroot

    return [
        platform_common.ToolchainInfo(
            name = ctx.label.name,
            compiler_executable = compiler_executable,
            all_files = toolchain_files,
            selectables_info = selectables_info,
            artifact_name_patterns = artifact_name_patterns,
            cuda_toolkit = cuda_toolchain_config.cuda_toolkit,
            **optional_attributes
        ),
    ]

cuda_toolchain = rule(
    doc = """This rule consumes a `CudaToolchainConfigInfo` and provides a `platform_common.ToolchainInfo`, a.k.a, the CUDA Toolchain.""",
    implementation = _cuda_toolchain_impl,
    toolchains = use_cpp_toolchain(),
    attrs = {
        "toolchain_config": attr.label(
            mandatory = True,
            providers = [CudaToolchainConfigInfo],
            doc = "A target that provides a `CudaToolchainConfigInfo`.",
        ),
        "compiler_use_cc_toolchain": attr.bool(default = False, doc = "Use existing cc_toolchain if configured as the compiler executable. Overrides compiler_executable or compiler_label"),
        "compiler_executable": attr.string(doc = "The path of the main executable of this toolchain. Either compiler_executable or compiler_label must be specified if compiler_use_cc_toolchain is not set."),
        "compiler_label": attr.label(allow_single_file = True, executable = True, cfg = "exec", doc = "The label of the main executable of this toolchain. Either compiler_executable or compiler_label must be specified."),
        "compiler_files": attr.label_list(allow_files = True, cfg = "exec", doc = "The set of files that are needed when compiling using this toolchain."),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    fragments = ["cpp"],
)

CUDA_TOOLCHAIN_TYPE = "//cuda:toolchain_type"

# buildifier: disable=unused-variable
def use_cuda_toolchain():
    """Helper to depend on the CUDA toolchain."""
    return [CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolchain(ctx):
    """Helper to get the cuda toolchain from context object.

    Args:
        ctx: The rule context for which to find a toolchain.

    Returns:
        A `platform_common.ToolchainInfo` that wraps around the necessary information of a cuda toolchain.
    """
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolkit(ctx):
    """Finds the CUDA toolchain.

    Args:
        ctx: The rule context for which to find a toolchain.

    Returns:
        A CudaToolkitInfo.
    """
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE].cuda_toolkit[CudaToolkitInfo]

# buildifier: disable=unnamed-macro
def register_detected_cuda_toolchains():
    """Helper to register the automatically detected CUDA toolchain(s).

User can setup their own toolchain if needed and ignore the detected ones by not calling this macro.
"""
    native.register_toolchains(
        "@cuda//toolchain:nvcc-local-toolchain",
        "@cuda//toolchain/clang:clang-local-toolchain",
        "@cuda//toolchain/disabled:disabled-local-toolchain",
    )
