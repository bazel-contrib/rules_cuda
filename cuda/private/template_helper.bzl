load("//cuda/private:redist_json_helper.bzl", "redist_json_helper")
load("//cuda/private:templates/registry.bzl", "REGISTRY")

def _to_forward_slash(s):
    return s.replace("\\", "/")

def _is_linux(ctx):
    return ctx.os.name.startswith("linux")

def _is_windows(ctx):
    return ctx.os.name.lower().startswith("windows")

def _expand_template(repository_ctx, tpl_label, substitutions):
    template_content = "# Generated from fragment " + str(tpl_label) + "\n"
    template_content += repository_ctx.read(tpl_label)

    for k, v in substitutions.items():
        template_content = template_content.replace(k, v)
    return template_content

def _version_label(version):
    return version.replace(".", "_").replace("-", "_")

def _major_minor(version):
    """Split a version into its (major, minor) pair, or None if it has neither.

    Args:
        version: A version string such as "12.8.1" or "11.5.1-1ubuntu1".

    Returns:
        A (major, minor) tuple of strings, or None if `version` is not of that shape.
    """
    parts = version.split("-", 1)[0].split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return (parts[0], parts[1])

def _versions_to_select_on(toolkit_versions):
    """The versions a toolchain attribute can be selected on.

    A single version needs no select at all, and one whose major/minor cannot be parsed
    has nothing to compare against `--@rules_cuda//cuda:version`, so both are dropped.

    Args:
        toolkit_versions: Every version declared with `cuda.redist_json`.

    Returns:
        The versions to emit a select branch for, or [] to keep the attribute a literal.
    """
    if not toolkit_versions or len(toolkit_versions) < 2:
        return []
    return [v for v in toolkit_versions if _major_minor(v) != None]

def _version_config_settings(toolkit_versions):
    """config_settings matching `--@rules_cuda//cuda:version`, one per declared version.

    Emitted into the same package as the selects that read them.

    Args:
        toolkit_versions: The versions returned by `_versions_to_select_on`.

    Returns:
        The config_setting definitions, or "" when no select is needed.
    """
    if not toolkit_versions:
        return ""

    lines = []
    for version in toolkit_versions:
        lines.extend([
            "config_setting(",
            '    name = "toolkit_version_is_{}",'.format(_version_label(version)),
            '    flag_values = {{"@rules_cuda//cuda:version": "{}"}},'.format(version),
            ")",
            "",
        ])
    return "\n".join(lines).rstrip("\n")

def _substitute_version_config_settings(substitutions, toolkit_versions):
    """Add the config_setting substitution, keyed so an empty value leaves no blank line.

    The placeholder sits on its own line in the template. Substituting an empty string
    would leave that line blank, so when there is nothing to emit the surrounding newlines
    are consumed along with it.

    Args:
        substitutions: The substitution dict to add to.
        toolkit_versions: The versions returned by `_versions_to_select_on`.
    """
    config_settings = _version_config_settings(toolkit_versions)
    if config_settings:
        substitutions["# %{toolkit_version_config_settings}"] = config_settings
    else:
        substitutions["\n# %{toolkit_version_config_settings}\n"] = ""

def _version_select(attr_name, toolkit_versions, value_of_version, default_value):
    """An attribute assignment selecting its value on the active CUDA version.

    Args:
        attr_name: Name of the attribute to assign.
        toolkit_versions: The versions returned by `_versions_to_select_on`.
        value_of_version: Callback mapping a version to its already-quoted attribute value.
        default_value: Already-quoted value used when `cuda:version` is unset, which
            leaves the toolchain reporting the same version the component aliases
            resolve to by default (the maximum declared one).

    Returns:
        Either `attr = <default>,` or `attr = select({...}),`.
    """
    if not toolkit_versions:
        return "{} = {},".format(attr_name, default_value)

    lines = ["{} = select({{".format(attr_name)]
    for version in toolkit_versions:
        lines.append('    ":toolkit_version_is_{}": {},'.format(
            _version_label(version),
            value_of_version(version),
        ))
    lines.append('    "//conditions:default": {},'.format(default_value))
    lines.append("}),")
    return "\n    ".join(lines)

def _expand_lctk_cuda(repository_ctx, components):
    tpl_label = Label("//cuda/private:templates/BUILD.lctk_cuda")
    substitutions = {
        "%{component_name}": "cuda",
    }
    return _expand_template(repository_ctx, tpl_label, substitutions = substitutions)

def _expand_dctk_cuda(repository_ctx, components):
    tpl_label = Label("//cuda/private:templates/BUILD.dctk_cuda")

    # Filter out components with empty REGISTRY entries
    valid_components = [comp for comp in components.keys() if len(REGISTRY[comp]) > 0]

    all_files_srcs_line = [comp + "_all_files" for comp in valid_components]
    all_files_srcs_line = "srcs = " + repr(all_files_srcs_line)

    license_srcs_line = [comp + "_license" for comp in valid_components]
    license_srcs_line = "srcs = " + repr(license_srcs_line)

    headers_deps_line = [comp + "_headers" for comp in valid_components]
    headers_deps_line = "deps = " + repr(headers_deps_line)

    substitutions = {
        "# %{dctk_cuda_all_files_srcs_line}": all_files_srcs_line,
        "# %{dctk_cuda_license_srcs_line}": license_srcs_line,
        "# %{dctk_cuda_headers_deps_line}": headers_deps_line,
    }
    return _expand_template(repository_ctx, tpl_label, substitutions = substitutions)

def _expand_dctk_component(repository_ctx, component):
    tpl_label = Label("//cuda/private:templates/BUILD.dctk_comp")
    substitutions = {
        "%{component_name}": component,
    }
    return _expand_template(repository_ctx, tpl_label, substitutions = substitutions)

def _device_runtime_static_libs_line(cuda):
    by_os = cuda.device_runtime_static_libs_by_os
    if by_os == None:
        return "device_runtime_static_libs = " + repr(cuda.device_runtime_static_libs_labels) + ","

    return "\n".join([
        "device_runtime_static_libs = select({",
        '    "@platforms//os:linux": {},'.format(repr(by_os["linux"])),
        '    "@platforms//os:windows": {},'.format(repr(by_os["windows"])),
        '    "//conditions:default": [],',
        "}),",
    ])

def _component_owns_cuda_repo_alias(component, target, components):
    if target == "culibos_a" and "culibos" in components:
        return component == "culibos"
    return True

def _generate_build_impl(repository_ctx, libpath, components, is_cuda_repo, is_deliverable):
    # stitch template fragment
    fragments = [
        Label("//cuda/private:templates/BUILD.cuda_shared"),
        Label("//cuda/private:templates/BUILD.cuda_build_setting"),
    ]
    if is_cuda_repo and not is_deliverable:  # generate `@cuda//BUILD` for local host CTK
        fragments.append(_expand_lctk_cuda(repository_ctx, components))
        fragments.extend([Label("//cuda/private:templates/BUILD.{}".format(c)) for c in components])
    elif is_cuda_repo and is_deliverable:  # generate `@cuda//BUILD` for CTK with deliverables
        fragments.append(_expand_dctk_cuda(repository_ctx, components))
    elif not is_cuda_repo and is_deliverable:  # generate `@cuda_<component>//BUILD` for a deliverable
        if len(components) != 1:
            fail("one deliverable at a time")
        comp = components.keys()[0]
        fragments.append(_expand_dctk_component(repository_ctx, comp))
        fragments.append(Label("//cuda/private:templates/BUILD.{}".format(comp)))
    else:
        fail("unreachable")

    template_content = []
    for frag in fragments:
        if type(frag) == type(""):
            template_content.append(frag)
        else:
            template_content.append("# Generated from fragment " + str(frag))
            template_content.append(repository_ctx.read(frag))

    if is_cuda_repo and is_deliverable:  # generate `@cuda//BUILD` for CTK with deliverables
        for comp in components:
            for target in REGISTRY[comp]:
                if not _component_owns_cuda_repo_alias(comp, target, components):
                    continue
                repo = components[comp]
                line = 'alias(name = "{target}", actual = "{repo}//:{target}")'.format(target = target, repo = repo)
                template_content.append(line)

            # add an empty line to separate aliased targets from different components
            template_content.append("")

    template_content = "\n".join(template_content)

    template_path = repository_ctx.path("BUILD.tpl")
    repository_ctx.file(template_path, content = template_content, executable = False)

    substitutions = {
        "%{component_name}": "cuda" if is_cuda_repo else components.keys()[0],
        "%{libpath}": libpath,
    }
    repository_ctx.template("BUILD", template_path, substitutions = substitutions, executable = False)

def _generate_build(repository_ctx, libpath, components = None, is_cuda_repo = True, is_deliverable = False):
    """Generate `@cuda//BUILD` or `@cuda_<component>//BUILD`

    Notes:
        - is_cuda_repo==False and is_deliverable==False is an error
        - is_cuda_repo==True  and is_deliverable==False generate `@cuda//BUILD` for local host CTK
        - is_cuda_repo==True  and is_deliverable==True  generate `@cuda//BUILD` for CTK with deliverables
        - is_cuda_repo==False and is_deliverable==True  generate `@cuda_<component>//BUILD` for a deliverable

    Args:
        repository_ctx: repository_ctx
        libpath: substitution of %{libpath}
        components: dict[str, str], the components of CTK to be included, mapped to the repo names for the components
        is_cuda_repo: See Notes, True for @cuda generation, False for @cuda_<component> generation.
        is_deliverable: See Notes
    """

    if is_cuda_repo and not is_deliverable:
        if components == None:
            components = [c for c in REGISTRY if len(REGISTRY[c]) > 0]
        else:
            for c in components:
                if c not in REGISTRY:
                    fail("{} is not a valid component")

    _generate_build_impl(repository_ctx, libpath, components, is_cuda_repo, is_deliverable)

def _generate_defs_bzl(repository_ctx, version_major, version_minor, is_local_ctk):
    tpl_label = Label("//cuda/private:templates/defs.bzl.tpl")
    substitutions = {
        "%{version_major}": str(version_major),
        "%{version_minor}": str(version_minor),
        "%{is_local_ctk}": str(is_local_ctk),
    }
    repository_ctx.template("defs.bzl", tpl_label, substitutions = substitutions, executable = False)

def _generate_redist_bzl(repository_ctx, component_specs, redist_version):
    """Generate `@rules_cuda_redist_json//:redist.bzl`

    Args:
        repository_ctx: repository_ctx
        component_specs: list of dict, dict keys are component_name, urls, sha256, strip_prefix and version
        redist_version: str
    """

    rules_cuda_components_body = []
    mapping = {}

    component_tpl = """cuda_component(
        name = "{repo_name}",
        component_name = "{component_name}",
        descriptive_name = "{descriptive_name}",
        sha256 = {sha256},
        strip_prefix = {strip_prefix},
        urls = {urls},
        version = "{version}",
    )"""

    for spec in component_specs:
        repo_name = redist_json_helper.get_repo_name(repository_ctx, spec)

        rules_cuda_components_body.append(
            component_tpl.format(
                repo_name = repo_name,
                component_name = spec["component_name"],
                descriptive_name = spec["descriptive_name"],
                sha256 = repr(spec["sha256"]),
                strip_prefix = repr(spec["strip_prefix"]),
                urls = repr(spec["urls"]),
                version = spec["version"],
            ),
        )
        mapping[spec["component_name"]] = "@" + repo_name

    tpl_label = Label("//cuda/private:templates/redist.bzl.tpl")
    substitutions = {
        "%{rules_cuda_components_body}": "\n\n    ".join(rules_cuda_components_body),
        "%{components_mapping}": repr(mapping),
        "%{version}": redist_version,
    }
    repository_ctx.template("redist.bzl", tpl_label, substitutions = substitutions, executable = False)

def _generate_toolchain_build(repository_ctx, cuda):
    # Hermetic/deliverable toolkits (no single local install path) register both
    # Linux and Windows nvcc toolchains so the client host OS does not prevent
    # resolving a CUDA toolchain for a different target/exec platform (e.g.
    # Windows Bazel client + Linux remote exec + linux-sbsa target).
    is_deliverable = cuda.path == None
    if is_deliverable:
        tpl_label = Label("//cuda/private:templates/BUILD.toolchain_deliverable")
    else:
        tpl_label = Label(
            "//cuda/private:templates/BUILD.toolchain_" +
            ("nvcc" if _is_linux(repository_ctx) else "nvcc_msvc"),
        )
    compiler_files = ["@cuda//:compiler_deps"]
    if cuda.cicc_label != None:
        compiler_files.append(cuda.cicc_label)
    if cuda.libdevice_label != None:
        compiler_files.append(cuda.libdevice_label)
    compiler_files_line = "compiler_files = " + repr(compiler_files) + ","
    device_runtime_static_libs_line = _device_runtime_static_libs_line(cuda)

    select_versions = _versions_to_select_on(repository_ctx.attr.toolkit_versions)
    substitutions = {
        "# %{compiler_files_line}": compiler_files_line,
        "# %{device_runtime_static_libs_line}": device_runtime_static_libs_line,
        "%{cuda_path}": _to_forward_slash(cuda.path) if cuda.path else "cuda-not-found",
        "# %{cuda_version_line}": _version_select(
            "version",
            select_versions,
            lambda v: '"{}.{}"'.format(*_major_minor(v)),
            '"{}.{}"'.format(cuda.version_major, cuda.version_minor),
        ),
        "# %{nvcc_version_major_line}": _version_select(
            "nvcc_version_major",
            select_versions,
            lambda v: _major_minor(v)[0],
            str(cuda.nvcc_version_major),
        ),
        "# %{nvcc_version_minor_line}": _version_select(
            "nvcc_version_minor",
            select_versions,
            lambda v: _major_minor(v)[1],
            str(cuda.nvcc_version_minor),
        ),
        "%{nvcc_label}": cuda.nvcc_label,
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
        "%{ptxas_label}": cuda.ptxas_label,
        # Host-default alias for register_detected_cuda_toolchains().
        "%{nvcc_local_actual}": "nvcc-linux-toolchain" if _is_linux(repository_ctx) else "nvcc-windows-toolchain",
        # msvc template needs a tmp path; deliverable Windows config does too.
        "%{env_tmp}": "C:/Windows/Temp",
    }
    if cuda.cicc_label:
        substitutions["# %{cicc_line}"] = "cicc = " + repr(cuda.cicc_label)
    if cuda.libdevice_label:
        substitutions["# %{libdevice_line}"] = "libdevice = " + repr(cuda.libdevice_label)
    _substitute_version_config_settings(substitutions, select_versions)

    env_tmp = repository_ctx.os.environ.get("TMP", repository_ctx.os.environ.get("TEMP", None))
    if env_tmp != None:
        substitutions["%{env_tmp}"] = _to_forward_slash(env_tmp)
    repository_ctx.template("toolchain/BUILD", tpl_label, substitutions = substitutions, executable = False)

def _generate_toolchain_clang_build(repository_ctx, cuda, clang_path_or_label):
    tpl_label = Label("//cuda/private:templates/BUILD.toolchain_clang")
    compiler_attr_line = ""
    clang_path_for_subst = ""
    clang_label_for_subst = ""

    compiler_use_cc_toolchain_env = repository_ctx.os.environ.get("CUDA_COMPILER_USE_CC_TOOLCHAIN", "false")
    if compiler_use_cc_toolchain_env == "true":
        compiler_attr_line = "compiler_use_cc_toolchain = True,"
    elif clang_path_or_label != None and (clang_path_or_label.startswith("//") or clang_path_or_label.startswith("@")):
        # Use compiler_label
        compiler_attr_line = 'compiler_label = "%{{clang_label}}",'
        clang_label_for_subst = clang_path_or_label
    else:
        # Use compiler_executable
        compiler_attr_line = 'compiler_executable = "%{{clang_path}}",'
        clang_path_for_subst = _to_forward_slash(clang_path_or_label) if clang_path_or_label else "cuda-clang-path-not-found"

    compiler_attr_line = compiler_attr_line.format(
        clang_label = "%{clang_label}",
        clang_path = "%{clang_path}",
    )

    compiler_files = []
    cuda_path_for_subst = ""
    path_data = None
    if cuda.path:  # local installation
        compiler_files.append("@cuda//:compiler_deps")
        cuda_path_for_subst = _to_forward_slash(cuda.path)
    else:  # scattered components
        cuda_path_for_subst = "$(location @cuda//:compiler_root)"
        path_data = ["@cuda//:compiler_root"]
        compiler_files.extend([
            "@cuda//:nvcc_all_files",
            "@cuda//:cccl_all_files",
            "@cuda//:cudart_all_files",
            "@cuda//:curand_all_files",
        ])
        if int(cuda.version_major) >= 13:
            compiler_files.extend([
                "@cuda//:nvvm_all_files",
            ])
    path_data_line = "path_data = " + repr(path_data) + ","
    compiler_files_line = "compiler_files = " + repr(compiler_files) + ","
    device_runtime_static_libs_line = _device_runtime_static_libs_line(cuda)

    select_versions = _versions_to_select_on(repository_ctx.attr.toolkit_versions)
    substitutions = {
        "# %{compiler_attribute_line}": compiler_attr_line,
        "# %{compiler_files_line}": compiler_files_line,
        "# %{device_runtime_static_libs_line}": device_runtime_static_libs_line,
        "%{clang_path}": clang_path_for_subst,  # Will be empty if label is used
        "%{clang_label}": clang_label_for_subst,  # Will be empty if path is used
        "%{cuda_path}": cuda_path_for_subst,
        "# %{path_data_line}": path_data_line,
        "# %{cuda_version_line}": _version_select(
            "version",
            select_versions,
            lambda v: '"{}.{}"'.format(*_major_minor(v)),
            '"{}.{}"'.format(cuda.version_major, cuda.version_minor),
        ),
        "%{nvcc_label}": cuda.nvcc_label,
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
        "%{ptxas_label}": cuda.ptxas_label,
    }
    if cuda.cicc_label:
        substitutions["# %{cicc_line}"] = "cicc = " + repr(cuda.cicc_label)
    if cuda.libdevice_label:
        substitutions["# %{libdevice_line}"] = "libdevice = " + repr(cuda.libdevice_label)
    _substitute_version_config_settings(substitutions, select_versions)

    if clang_label_for_subst:
        substitutions.pop("%{clang_path}")
    if clang_path_for_subst:
        substitutions.pop("%{clang_label}")

    repository_ctx.template(
        "toolchain/clang/BUILD",
        tpl_label,
        substitutions = substitutions,
        executable = False,
    )

template_helper = struct(
    generate_build = _generate_build,
    generate_defs_bzl = _generate_defs_bzl,
    generate_redist_bzl = _generate_redist_bzl,
    generate_toolchain_build = _generate_toolchain_build,
    generate_toolchain_clang_build = _generate_toolchain_clang_build,
)
