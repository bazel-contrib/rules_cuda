"""private helpers"""

ArchSpecInfo = provider(
    "",
    fields = {
        "stage1_arch": "A virtual architecture",
        "stage2_archs": "A list of virtual or gpu architecture",
    },
)

CUDA_SRC_FILES = [".cu", ".cu.cc"]
DLINK_OBJ_FILES = [""]

def _get_arch_number(arch_str):
    arch_str = arch_str.strip()
    arch_num = None
    if arch_str.startswith("compute_"):
        arch_num = arch_str[len("compute_"):]
    if arch_str.startswith("sm_"):
        arch_num = arch_str[len("sm_"):]
    return int(arch_num)

def _get_arch_specs(specs_str):
    virtual_codes_pairs = [spec_str.strip().split(":") for spec_str in specs_str.strip().split(";")]
    archs = []
    for virtual_codes in virtual_codes_pairs:
        virt = None
        codes = None
        if len(virtual_codes) == 2:
            virt, codes = virtual_codes
        else:
            codes = virtual_codes
            virt = "compute_" + str(min([_get_arch_number(c) for c in codes]))
        archs.append(ArchSpecInfo(stage1_arch = virt, stage2_archs = virt))

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

def _get_basename_without_ext(basename, allow_exts, fail_if_not_match=True):
    for ext in sorted(allow_exts, key=len, reverse=True):
        if basename.endswith(ext):
            return basename[:-len(ext)]
    if fail_if_not_match:
        fail("'{}' does not have valide extension, allowed extension(s): {}".format(basename, allow_exts))
    else:
        return None

cuda_helper = struct(
    get_arch_number = _get_arch_number,
    get_arch_specs = _get_arch_specs,
    check_src_extension = _check_src_extension,
    check_srcs_extensions = _check_srcs_extensions,
    get_basename_without_ext = _get_basename_without_ext,
)
