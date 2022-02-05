"""private helpers"""

load("//cuda/private:providers.bzl", "ArchSpecInfo", "cuda_archs")

CUDA_SRC_FILES = [".cu", ".cu.cc"]
DLINK_OBJ_FILES = [""]

def _get_arch_number(arch_str):
    arch_str = arch_str.strip()
    arch_num = None
    if arch_str.startswith("compute_"):
        arch_num = arch_str[len("compute_"):]
    elif arch_str.startswith("compute_"):
        arch_num = arch_str[len("compute_"):]
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
        fail("'{}' does not have valide extension, allowed extension(s): {}".format(basename, allow_exts))
    else:
        return None

cuda_helper = struct(
    get_arch_number = _get_arch_number,
    get_arch_spec = _get_arch_spec,
    get_arch_specs = _get_arch_specs,
    check_src_extension = _check_src_extension,
    check_srcs_extensions = _check_srcs_extensions,
    get_basename_without_ext = _get_basename_without_ext,
)
