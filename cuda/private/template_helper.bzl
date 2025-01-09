load("//cuda/private:templates/registry.bzl", "REGISTRY")

def _to_forward_slash(s):
    return s.replace("\\", "/")

def _is_linux(ctx):
    return ctx.os.name.startswith("linux")

def _is_windows(ctx):
    return ctx.os.name.lower().startswith("windows")

def _generate_build(repository_ctx, libpath):
    # stitch template fragment
    fragments = [
        Label("//cuda/private:templates/BUILD.local_cuda_shared"),
        Label("//cuda/private:templates/BUILD.local_cuda_headers"),
        Label("//cuda/private:templates/BUILD.local_cuda_build_setting"),
    ]
    fragments.extend([Label("//cuda/private:templates/BUILD.{}".format(c)) for c in REGISTRY if len(REGISTRY[c]) > 0])

    template_content = []
    for frag in fragments:
        template_content.append("# Generated from fragment " + str(frag))
        template_content.append(repository_ctx.read(frag))

    template_content = "\n".join(template_content)

    template_path = repository_ctx.path("BUILD.tpl")
    repository_ctx.file(template_path, content = template_content, executable = False)

    substitutions = {
        "%{component_name}": "cuda",
        "%{libpath}": libpath,
    }
    repository_ctx.template("BUILD", template_path, substitutions = substitutions, executable = False)

def _generate_defs_bzl(repository_ctx, is_local_cuda):
    tpl_label = Label("//cuda/private:templates/defs.bzl.tpl")
    substitutions = {
        "%{is_local_cuda}": str(is_local_cuda),
    }
    repository_ctx.template("defs.bzl", tpl_label, substitutions = substitutions, executable = False)

def _generate_toolchain_build(repository_ctx, cuda):
    tpl_label = Label(
        "//cuda/private:templates/BUILD.local_toolchain_" +
        ("nvcc" if _is_linux(repository_ctx) else "nvcc_msvc"),
    )
    substitutions = {
        "%{cuda_path}": _to_forward_slash(cuda.path) if cuda.path else "cuda-not-found",
        "%{cuda_version}": "{}.{}".format(cuda.version_major, cuda.version_minor),
        "%{nvcc_version_major}": str(cuda.nvcc_version_major),
        "%{nvcc_version_minor}": str(cuda.nvcc_version_minor),
        "%{nvcc_label}": cuda.nvcc_label,
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
    }
    env_tmp = repository_ctx.os.environ.get("TMP", repository_ctx.os.environ.get("TEMP", None))
    if env_tmp != None:
        substitutions["%{env_tmp}"] = _to_forward_slash(env_tmp)
    repository_ctx.template("toolchain/BUILD", tpl_label, substitutions = substitutions, executable = False)

def _generate_toolchain_clang_build(repository_ctx, cuda, clang_path):
    tpl_label = Label("//cuda/private:templates/BUILD.local_toolchain_clang")
    substitutions = {
        "%{clang_path}": _to_forward_slash(clang_path) if clang_path else "cuda-clang-not-found",
        "%{cuda_path}": _to_forward_slash(cuda.path) if cuda.path else "cuda-not-found",
        "%{cuda_version}": "{}.{}".format(cuda.version_major, cuda.version_minor),
        "%{nvcc_label}": cuda.nvcc_label,
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
    }
    repository_ctx.template("toolchain/clang/BUILD", tpl_label, substitutions = substitutions, executable = False)

template_helper = struct(
    generate_build = _generate_build,
    generate_defs_bzl = _generate_defs_bzl,
    generate_toolchain_build = _generate_toolchain_build,
    generate_toolchain_clang_build = _generate_toolchain_clang_build,
)
