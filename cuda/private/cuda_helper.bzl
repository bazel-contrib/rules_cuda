"""private helpers"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:artifact_categories.bzl", "ARTIFACT_CATEGORIES")
load("//cuda/private:providers.bzl", "ArchSpecInfo", "CudaArchsInfo", "CudaInfo", "Stage2ArchInfo", "cuda_archs")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper")

def _get_arch_number(arch_str):
    arch_str = arch_str.strip()
    arch_num = None
    if arch_str.startswith("compute_"):
        arch_num = arch_str[len("compute_"):]
    elif arch_str.startswith("lto_"):
        arch_num = arch_str[len("lto_"):]
    elif arch_str.startswith("sm_"):
        arch_num = arch_str[len("sm_"):]
    if arch_num not in cuda_archs:
        fail("{} is not a supported cuda arch".format(arch_str))
    return int(arch_num)

def _get_stage2_arch_info(code_str):
    return Stage2ArchInfo(
        arch = str(_get_arch_number(code_str)),
        virtual = code_str.startswith("compute_"),
        gpu = code_str.startswith("sm_"),
        lto = code_str.startswith("lto_"),
    )

def _get_arch_spec(spec_str):
    '''Convert string into an ArchSpecInfo.

    aka, parse "compute_80:sm_80,sm_86"'''
    stage1_arch = None
    stage2_archs = []

    virt = None  # stage1 str
    codes = None  # stage2 str
    virtual_codes = spec_str.split(":")
    if len(virtual_codes) == 2:
        virt, codes = virtual_codes
        codes = codes.split(",")
        if not virt.startswith("compute_"):
            fail("expect a virtual architecture, got", virt)
        stage1_arch = str(_get_arch_number(virt))
        stage2_archs = [_get_stage2_arch_info(code) for code in codes]
    else:
        (codes,) = virtual_codes
        codes = codes.split(",")
        stage1_arch = str(min([_get_arch_number(c) for c in codes]))
        stage2_archs = [_get_stage2_arch_info(code) for code in codes]
    arch_spec = ArchSpecInfo(stage1_arch = stage1_arch, stage2_archs = stage2_archs)
    return arch_spec

def _get_arch_specs(specs_str):
    '''Convert string into a list of ArchSpecInfo.

    aka, parse "compute_70:sm_70;compute_80:sm_80,sm_86"'''
    archs = []
    for sepc_str in specs_str.split(";"):
        archs.append(_get_arch_spec(sepc_str))
    return archs

def _check_src_extension(file, allowed_src_files):
    for pattern in allowed_src_files:
        if file.basename.endswith(pattern):
            return True
    return False

def _check_srcs_extensions(ctx, allowed_src_files, rule_name):
    for src in ctx.attr.srcs:
        files = src[DefaultInfo].files.to_list()
        if len(files) == 1 and files[0].is_source:
            if not _check_src_extension(files[0], allowed_src_files) and not files[0].is_directory:
                fail("in srcs attribute of {} rule {}: source file '{}' is misplaced here".format(rule_name, ctx.label, str(src.label)))
        else:
            at_least_one_good = False
            for file in files:
                if _check_src_extension(file, allowed_src_files) or file.is_directory:
                    at_least_one_good = True
                    break
            if not at_least_one_good:
                fail("'{}' does not produce any {} srcs files".format(str(src.label), rule_name), attr = "srcs")

def _get_basename_without_ext(basename, allow_exts, fail_if_not_match = True):
    for ext in sorted(allow_exts, key = len, reverse = True):
        if basename.endswith(ext):
            return basename[:-len(ext)]
    if fail_if_not_match:
        fail("'{}' does not have valid extension, allowed extension(s): {}".format(basename, allow_exts))
    else:
        return None

# # TODO: Remove, impl use cuda_toolchain_config
# def _get_nvcc_compile_arch_flags(arch_specs):
#     tpl = "arch={},code={}"
#     ret = []
#     for arch_spec in arch_specs:
#         for stage2_arch in arch_spec.stage2_archs:
#             ret.append("-gencode")
#             ret.append(tpl.format(arch_spec.stage1_arch, stage2_arch))
#     return ret

# # TODO: Remove, impl use cuda_toolchain_config
# def _get_nvcc_dlink_arch_flags(arch_specs):
#     tpl = "arch={},code={}"
#     ret = []
#     lto = False
#     for arch_spec in arch_specs:
#         for stage2_arch in arch_spec.stage2_archs:
#             # https://forums.developer.nvidia.com/t/using-dlink-time-opt-together-with-gencode-in-cmake/165224/4
#             if stage2_arch.startswith("lto_"):
#                 lto = True
#                 stage2_arch = stage2_arch.replace("lto_", "sm_", 1)
#             ret.append("-gencode")
#             ret.append(tpl.format(arch_spec.stage1_arch, stage2_arch))
#     if lto:
#         ret.append("-dlto")
#     return ret

def _get_clang_arch_flags(arch_specs):
    fail("not implemented")

def _resolve_includes(ctx, path):
    if paths.is_absolute(path):
        fail("invalid absolute path", path)

    src_path = paths.normalize(paths.join(ctx.label.workspace_root, ctx.label.package, path))
    bin_path = paths.join(ctx.bin_dir.path, src_path)
    return src_path, bin_path

def _check_opts(opt):
    opt = opt.strip()
    if (opt.startswith("--generate-code") or opt.startswith("-gencode") or
        opt.startswith("--gpu-architecture") or opt.startswith("-arch") or
        opt.startswith("--gpu-code") or opt.startswith("-code") or
        opt.startswith("--relocatable-device-code") or opt.startswith("-rdc") or
        opt.startswith("--cuda") or opt.startswith("-cuda") or
        opt.startswith("--preprocess") or opt.startswith("-E") or
        opt.startswith("--compile") or opt.startswith("-c") or
        opt.startswith("--cubin") or opt.startswith("-cubin") or
        opt.startswith("--ptx") or opt.startswith("-ptx") or
        opt.startswith("--fatbin") or opt.startswith("-fatbin") or
        opt.startswith("--device-link") or opt.startswith("-dlink") or
        opt.startswith("--lib") or opt.startswith("-lib") or
        opt.startswith("--generate-dependencies") or opt.startswith("-M") or
        opt.startswith("--generate-nonsystem-dependencies") or opt.startswith("-MM") or
        opt.startswith("--run") or opt.startswith("-run")):
        fail(opt, "is not allowed to be specified directly via copts")
    return True

def _create_common(ctx):
    attr = ctx.attr

    # gather include info
    includes = []
    system_includes = []
    quote_includes = []
    for inc in attr.includes:
        system_includes.extend(_resolve_includes(ctx, inc))
    for dep in attr.deps:
        if CcInfo in dep:
            includes.extend(dep[CcInfo].compilation_context.includes.to_list())
            system_includes.extend(dep[CcInfo].compilation_context.system_includes.to_list())
            quote_includes.extend(dep[CcInfo].compilation_context.quote_includes.to_list())

    # gather header info
    public_headers = []
    private_headers = []
    for fs in attr.hdrs:
        public_headers.extend(fs.files.to_list())
    for fs in attr.srcs:
        hdr = [f for f in fs.files.to_list() if cuda_helper.check_src_extension(f, ALLOW_CUDA_HDRS)]
        private_headers.extend(hdr)
    headers = public_headers + private_headers
    for dep in attr.deps:
        if CcInfo in dep:
            headers.extend(dep[CcInfo].compilation_context.headers.to_list())

    # gather linker info
    builtin_linker_inputs = []
    if hasattr(attr, "_builtin_deps"):
        builtin_linker_inputs = [dep[CcInfo].linking_context.linker_inputs for dep in attr._builtin_deps if CcInfo in dep]

    transitive_linker_inputs = [dep[CcInfo].linking_context.linker_inputs for dep in attr.deps if CcInfo in dep]

    # gather compile info
    defines = []
    local_defines = [i for i in attr.local_defines]
    compile_flags = [o for o in attr.copts if _check_opts(o)]
    link_flags = []
    if hasattr(attr, "linkopts"):
        link_flags.extend([o for o in attr.linkopts if _check_opts(o)])
    host_defines = []
    host_local_defines = [i for i in attr.host_local_defines]
    host_compile_flags = [i for i in attr.host_copts]
    host_link_flags = []
    if hasattr(attr, "host_linkopts"):
        host_link_flags.extend([i for i in attr.host_linkopts])
    for dep in attr.deps:
        if CudaInfo in dep:
            defines.extend(dep[CudaInfo].defines.to_list())
        if CcInfo in dep:
            host_defines.extend(dep[CcInfo].compilation_context.defines.to_list())
    defines.extend(attr.defines)
    host_defines.extend(attr.host_defines)

    return struct(
        includes = includes,
        quote_includes = quote_includes,
        system_includes = system_includes,
        headers = depset(headers),
        transitive_linker_inputs = builtin_linker_inputs + transitive_linker_inputs,
        defines = defines,
        local_defines = local_defines,
        compile_flags = compile_flags,
        link_flags = link_flags,
        host_defines = host_defines,
        host_local_defines = host_local_defines,
        host_compile_flags = host_compile_flags,
        host_link_flags = host_link_flags,
    )

def _create_cuda_info(defines = None, objects = None, rdc_objects = None, pic_objects = None, rdc_pic_objects = None):
    ret = CudaInfo(
        defines = defines if defines != None else depset([]),
        objects = objects if objects != None else depset([]),
        rdc_objects = rdc_objects if rdc_objects != None else depset([]),
        pic_objects = pic_objects if pic_objects != None else depset([]),
        rdc_pic_objects = rdc_pic_objects if rdc_pic_objects != None else depset([]),
    )
    return ret

def _get_artifact_category_from_action(action_name, use_pic = None, use_rdc = None):
    if action_name == ACTION_NAMES.cuda_compile:
        if use_pic:
            if use_rdc:
                return ARTIFACT_CATEGORIES.rdc_pic_object_file
            else:
                return ARTIFACT_CATEGORIES.pic_object_file
        elif use_rdc:
            return ARTIFACT_CATEGORIES.rdc_object_file
        else:
            return ARTIFACT_CATEGORIES.object_file
    elif action_name == ACTION_NAMES.device_link:
        if not use_rdc:
            fail("non relocatable device code cannot be device linked")
        if use_pic:
            return ARTIFACT_CATEGORIES.rdc_pic_object_file
        else:
            return ARTIFACT_CATEGORIES.rdc_object_file
    elif action_name == ACTION_NAMES.create_library:
        if use_pic:
            return ARTIFACT_CATEGORIES.pic_archive
        else:
            return ARTIFACT_CATEGORIES.archive
    else:
        fail("NotImplemented")

def _get_artifact_name(cuda_toolchain, category_name, output_basename):
    return config_helper.get_artifact_name(cuda_toolchain.artifact_name_patterns, category_name, output_basename)

def _check_must_enforce_rdc(*, arch_specs = None, cuda_archs_info = None):
    if arch_specs == None:
        arch_specs = cuda_archs_info.arch_specs
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.lto:
                return True
    return False

def _create_compile_variables(
        cuda_toolchain,
        feature_configuration,
        cuda_archs_info,
        source_file = None,
        output_file = None,
        host_compiler = None,
        user_compile_flags = [],
        include_paths = [],
        quote_include_paths = [],
        system_include_paths = [],
        defines = [],
        host_defines = [],
        use_pic = False,
        use_rdc = False):
    arch_specs = cuda_archs_info.arch_specs
    if not use_rdc:
        use_rdc = _check_must_enforce_rdc(arch_specs = arch_specs)

    return struct(
        arch_specs = arch_specs,
        source_file = source_file,
        output_file = output_file,
        host_compiler = host_compiler,
        user_compile_flags = user_compile_flags,
        include_paths = include_paths,
        quote_include_paths = quote_include_paths,
        system_include_paths = system_include_paths,
        defines = defines,
        host_defines = host_defines,
        use_pic = use_pic,
        use_rdc = use_rdc,
    )

def _create_device_link_variables(
        cuda_toolchain,
        feature_configuration,
        cuda_archs_info,
        output_file = None,
        host_compiler = None,
        library_search_paths = [],
        runtime_library_search_paths = [],
        user_link_flags = []):
    arch_specs = cuda_archs_info.arch_specs
    use_dlto = False
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.lto:
                use_lto = True
                break
    return struct(
        arch_specs = arch_specs,
        output_file = output_file,
        host_compiler = host_compiler,
        library_search_paths = library_search_paths,
        runtime_library_search_paths = runtime_library_search_paths,
        user_link_flags = user_link_flags,
        use_dlto = use_dlto,
    )

def _configure_features(ctx, cuda_toolchain, requested_features = None, unsupported_features = None):
    return config_helper.configure_features(
        selectables_info = cuda_toolchain.selectables_info,
        requested_features = requested_features,
        unsupported_features = unsupported_features,
    )

cuda_helper = struct(
    get_arch_specs = _get_arch_specs,
    check_src_extension = _check_src_extension,
    check_srcs_extensions = _check_srcs_extensions,
    check_must_enforce_rdc = _check_must_enforce_rdc,
    get_basename_without_ext = _get_basename_without_ext,
    create_common = _create_common,
    create_cuda_info = _create_cuda_info,
    get_artifact_category_from_action = _get_artifact_category_from_action,
    get_artifact_name = _get_artifact_name,
    create_compile_variables = _create_compile_variables,
    create_device_link_variables = _create_device_link_variables,
    configure_features = _configure_features,  # wrapped for collecting info from ctx and cuda_toolchain
    get_default_features_and_action_configs = config_helper.get_default_features_and_action_configs,
    get_enabled_feature = config_helper.get_enabled_feature,
    get_command_line = config_helper.get_command_line,
    get_tool_for_action = config_helper.get_tool_for_action,
    action_is_enabled = config_helper.is_enabled,
    is_enabled = config_helper.is_enabled,
    get_environment_variables = config_helper.get_environment_variables,

    # TODO: Remove or hide
    get_arch_number = _get_arch_number,
    get_arch_spec = _get_arch_spec,
)
