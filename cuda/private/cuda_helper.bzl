"""private helpers"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("//cuda/private:providers.bzl", "ArchSpecInfo", "cuda_archs", "CudaArchsInfo")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS")

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

def _get_arch_spec(spec_str):
    '''Convert string into an ArchSpecInfo.

    aka, parse "compute_80:sm_80,sm_86"'''
    virt = None
    codes = None
    virtual_codes = spec_str.split(":")
    if len(virtual_codes) == 2:
        virt, codes = virtual_codes
        codes = codes.split(",")
        check_invalid_arch = _get_arch_number(virt)
        check_invalid_arch = [_get_arch_number(code) for code in codes]
    else:
        (codes,) = virtual_codes
        codes = codes.split(",")
        virt = "compute_" + str(min([_get_arch_number(c) for c in codes]))
        check_invalid_arch = _get_arch_number(virt)
    arch_spec = ArchSpecInfo(stage1_arch = virt, stage2_archs = codes)
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

def _get_nvcc_compile_arch_flags(arch_specs):
    tpl = "arch={},code={}"
    ret = []
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            ret.append("-gencode")
            ret.append(tpl.format(arch_spec.stage1_arch, stage2_arch))
    return ret

def _get_nvcc_dlink_arch_flags(arch_specs):
    tpl = "arch={},code={}"
    ret = []
    lto = False
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            # https://forums.developer.nvidia.com/t/using-dlink-time-opt-together-with-gencode-in-cmake/165224/4
            if stage2_arch.startswith("lto_"):
                lto = True
                stage2_arch = stage2_arch.replace("lto_", "sm_", 1)
            ret.append("-gencode")
            ret.append(tpl.format(arch_spec.stage1_arch, stage2_arch))
    if lto:
        ret.append("-dlto")
    return ret

def _get_clang_arch_flags(arch_specs):
    fail("not implemented")

def _resolve_includes(ctx, path):
    if paths.is_absolute(path):
        fail("invalid absolute path", path)

    src_path = paths.normalize(paths.join(ctx.label.workspace_root, ctx.label.package, path))
    bin_path = paths.join(ctx.bin_dir.path, src_path)
    return src_path, bin_path

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
    local_defines = []
    compile_flags = _get_nvcc_compile_arch_flags(attr._default_cuda_archs[CudaArchsInfo].arch_specs)
    link_flags = _get_nvcc_dlink_arch_flags(attr._default_cuda_archs[CudaArchsInfo].arch_specs)
    host_defines = []
    host_local_defines = []
    host_compile_flags = []
    host_link_flags = []

    return struct(
        includes = depset(includes),
        system_includes = depset(system_includes),
        quote_includes = depset(quote_includes),
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

cuda_helper = struct(
    get_arch_number = _get_arch_number,
    get_arch_spec = _get_arch_spec,
    get_arch_specs = _get_arch_specs,
    check_src_extension = _check_src_extension,
    check_srcs_extensions = _check_srcs_extensions,
    get_basename_without_ext = _get_basename_without_ext,
    create_common = _create_common,
)
