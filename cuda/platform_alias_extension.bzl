"""Module extension for creating platform-specific aliases for external dependencies.

This extension creates repositories with alias targets that select between x86_64 and ARM64
repositories based on the build platform. It also selects between versions of the component.
"""

load("//cuda/private:platforms.bzl", "SUPPORTED_PLATFORMS")
load("//cuda/private:templates/registry.bzl", "REGISTRY")

# Use REGISTRY as the source of truth for component targets
TARGET_MAPPING = REGISTRY

# Mapping from @platforms-constraint config_settings (in @rules_cuda//cuda)
# to the canonical CUDA platform they should resolve to. Covers only the
# platforms unambiguously identified by @platforms//cpu + @platforms//os
# alone. linux-sbsa and linux-aarch64 share aarch64+linux constraints and
# are routed via :is_linux_aarch64_or_sbsa to a nested alias keyed on the
# :target_gpu flag.
#
# The constraint check evaluates against the *active configuration's* platform
# (exec at cfg="exec", target otherwise) — RBE-safe and reuses Bazel's normal
# platform resolution.
#
# Adding a new platform to SUPPORTED_PLATFORMS requires three matching changes:
#   1. A new `is_<platform>` config_setting in //cuda:BUILD.bazel
#   2. If the new platform shares cpu+os with another, a disambiguation
#      mechanism (see :target_gpu and :is_linux_aarch64_or_sbsa for the
#      aarch64 case)
#   3. A new entry in this list (or in _AARCH64_FROM_TARGET_GPU below for
#      aarch64-only additions)
# Without all three, the new platform will silently fall through to
# :unsupported_cuda_platform with no diagnostic.
_AUTO_FROM_CONSTRAINT = [
    ("is_linux_x86_64", "linux-x86_64"),
    ("is_windows_x86_64", "windows-x86_64"),
]

# Mapping from :target_gpu-flag config_settings to the aarch64-linux CUDA
# platform they resolve to. Drives the inner alias that disambiguates
# linux-sbsa (server-class ARM, discrete GPU) from linux-aarch64 (Tegra/Drive
# embedded, on-die GPU).
_AARCH64_FROM_TARGET_GPU = [
    ("target_gpu_is_discrete", "linux-sbsa"),
    ("target_gpu_is_on_die", "linux-aarch64"),
]

def _platform_repos_attr(platform):
    return platform.replace("-", "_") + "_repos"

_PLATFORM_REPO_ATTRS = {
    _platform_repos_attr(_platform): attr.string_dict(
        default = {},
        doc = "Dictionary mapping versions to repository names for {}".format(_platform),
    )
    for _platform in SUPPORTED_PLATFORMS
}

def _version_sort_key(version):
    prefix = version.split("-", 1)[0]
    parts = prefix.split(".")
    if all([p.isdigit() for p in parts]):
        return (1, [int(p) for p in parts], version)
    return (0, [], version)

def _emit_aarch64_inner_alias(build_content, alias_name, target_name, platforms_available, dummy_target):
    """Emit the private inner alias that disambiguates linux-sbsa vs linux-aarch64.

    Keyed on the :target_gpu flag (discrete -> linux-sbsa, on_die ->
    linux-aarch64). Referenced from the outer constraint-based alias when
    the active config's platform is aarch64-linux.
    """
    build_content.append("alias(")
    build_content.append('    name = "{}",'.format(alias_name))
    build_content.append("    actual = select({")
    for setting, platform in _AARCH64_FROM_TARGET_GPU:
        platform_suffix = platform.replace("-", "_")
        build_content.append('        "@rules_cuda//cuda:{}":'.format(setting))
        if platform in platforms_available:
            build_content.append('            ":{}_{}",'.format(platform_suffix, target_name))
        else:
            build_content.append('            "{}",'.format(dummy_target))
    build_content.append("    }),")
    build_content.append('    visibility = ["//visibility:private"],')
    build_content.append(")")
    build_content.append("")

def _emit_outer_constraint_alias(build_content, alias_name, target_name, platforms_available, dummy_target, visibility, aarch64_inner_name):
    """Emit the alias whose actual is the @platforms-constraint platform select.

    Used directly for target components and as the auto-detection fallback
    for exec components. For aarch64-linux configs, routes to the private
    :target_gpu-keyed inner alias passed as aarch64_inner_name. Constraint
    check fires against the active config's platform (exec at cfg="exec",
    target otherwise) — RBE-safe.
    """
    build_content.append("alias(")
    build_content.append('    name = "{}",'.format(alias_name))
    build_content.append("    actual = select({")
    for setting, platform in _AUTO_FROM_CONSTRAINT:
        platform_suffix = platform.replace("-", "_")
        build_content.append('        "@rules_cuda//cuda:{}":'.format(setting))
        if platform in platforms_available:
            build_content.append('            ":{}_{}",'.format(platform_suffix, target_name))
        else:
            build_content.append('            "{}",'.format(dummy_target))
    build_content.append('        "@rules_cuda//cuda:is_linux_aarch64_or_sbsa":')
    build_content.append('            ":{}",'.format(aarch64_inner_name))
    build_content.append('        "//conditions:default": ":unsupported_cuda_platform",')
    build_content.append("    }),")
    build_content.append('    visibility = ["{}"],'.format(visibility))
    build_content.append(")")
    build_content.append("")

def _platform_alias_repo_impl(ctx):
    """Implementation of the platform_alias_repo repository rule.

    Args:
        ctx: Repository context with attributes x86_repo, arm64_repo, and targets.
    """

    # Generate BUILD.bazel content with platform-specific aliases
    build_content = ["# Generated by platform_alias_repo rule", ""]

    # Add load statement for alias
    build_content.append('load("@rules_cuda//cuda:defs.bzl", "unsupported_cuda_version", "unsupported_cuda_platform")')
    build_content.append("")

    build_content.append("[")
    build_content.append("    config_setting(")
    build_content.append('        name = "version_is_{}_" + version.replace(".", "_"),'.format(
        ctx.attr.component_name,
    ))
    build_content.append('        flag_values = {"@rules_cuda//cuda:version": "{}".format(version)},')
    build_content.append("    )")
    build_content.append("    for version in {}".format(ctx.attr.versions))
    build_content.append("]")
    build_content.append("")

    build_content.append(
        'unsupported_cuda_version(name = "unsupported_cuda_version", component = "{}", available_versions = {})'.format(
            ctx.attr.component_name,
            ctx.attr.versions,
        ),
    )
    build_content.append("")

    # Build a target for the name of the repo (only if at least one platform is available).
    platform_type = "exec" if ctx.attr.component_name in ["nvcc", "nvvm"] else "target"

    platform_repos_map = {
        platform: getattr(ctx.attr, _platform_repos_attr(platform))
        for platform in SUPPORTED_PLATFORMS
    }

    # Check which platforms are available (have at least one version).
    platforms_available = [platform for platform in SUPPORTED_PLATFORMS if len(platform_repos_map[platform]) > 0]

    # Always create unsupported_cuda_platform target - it's used as the default case
    # in select() when no platform condition matches.
    build_content.append(
        'unsupported_cuda_platform(name = "unsupported_cuda_platform", component = "{}", available_platforms = {})'.format(
            ctx.attr.component_name,
            platforms_available,
        ),
    )
    build_content.append("")

    # Only generate target aliases if this component is in TARGET_MAPPING.
    if ctx.attr.component_name not in TARGET_MAPPING:
        # Write the BUILD.bazel file with just the main alias.
        ctx.file("BUILD.bazel", "\n".join(build_content))
        return

    for target in TARGET_MAPPING[ctx.attr.component_name]:
        # Create alias for each target with platform selection.
        # Always add conditions for ALL platforms so that builds on any platform
        # have a matching select condition. Platforms where the component doesn't
        # exist will use a dummy target.
        target_name = target if target.find("/") == -1 else target.split("/")[-1]

        # Determine appropriate dummy target based on the target name.
        dummy_target = "@rules_cuda//cuda/dummy:dummy"
        if target_name == "cicc":
            dummy_target = "@rules_cuda//cuda/dummy:cicc"
        elif target_name == "libdevice.10.bc":
            dummy_target = "@rules_cuda//cuda/dummy:libdevice.10.bc"

        # Inner aarch64 disambiguation alias, shared by the target / exec
        # outer aliases when the active config's platform is aarch64-linux.
        aarch64_inner_name = "aarch64_or_sbsa_" + target_name
        _emit_aarch64_inner_alias(
            build_content,
            aarch64_inner_name,
            target_name,
            platforms_available,
            dummy_target,
        )

        if platform_type == "exec":
            # Outer alias: explicit --exec_platform flag wins; the default
            # branch falls through to constraint-based auto-detection in
            # the private :auto_<name> alias below.
            build_content.append("alias(")
            build_content.append('    name = "{}",'.format(target_name))
            build_content.append("    actual = select({")
            for platform in SUPPORTED_PLATFORMS:
                platform_suffix = platform.replace("-", "_")
                build_content.append(
                    '        "@rules_cuda//cuda:exec_platform_is_{}":'.format(platform_suffix),
                )
                if platform in platforms_available:
                    build_content.append('            ":{}_{}",'.format(platform_suffix, target_name))
                else:
                    build_content.append('            "{}",'.format(dummy_target))
            build_content.append('        "//conditions:default": ":auto_{}",'.format(target_name))
            build_content.append("    }),")
            build_content.append('    visibility = ["//visibility:public"],')
            build_content.append(")")
            build_content.append("")

            # Auto-detection fallback. Private — only the outer flag-select
            # in this same package needs to reach it.
            _emit_outer_constraint_alias(
                build_content,
                "auto_" + target_name,
                target_name,
                platforms_available,
                dummy_target,
                "//visibility:private",
                aarch64_inner_name,
            )
        else:
            # Target components: single outer alias keyed on @platforms
            # constraints. --platforms is the primary knob; :target_gpu
            # picks the aarch64 variant in the inner alias.
            _emit_outer_constraint_alias(
                build_content,
                target_name,
                target_name,
                platforms_available,
                dummy_target,
                "//visibility:public",
                aarch64_inner_name,
            )

        # Generate platform-specific aliases for ALL platforms.
        # Platforms where the component exists get version-based selection.
        # Platforms where it doesn't exist get dummy targets for all versions.
        # This ensures builds on any platform have matching select conditions.

        for platform in SUPPORTED_PLATFORMS:
            platform_suffix = platform.replace("-", "_")
            repos_dict = platform_repos_map[platform]
            platform_available = platform in platforms_available
            default_version = sorted(ctx.attr.versions, key = _version_sort_key)[-1] if ctx.attr.versions else None

            build_content.append("alias(")
            build_content.append('    name = "{}_{}",'.format(platform_suffix, target_name))
            build_content.append("    actual = select({")

            for version in ctx.attr.versions:
                build_content.append('        ":version_is_{}_{}": '.format(
                    ctx.attr.component_name,
                    version.replace(".", "_"),
                ))
                if platform_available and version in repos_dict:
                    repo_name = repos_dict[version]
                    build_content.append(
                        '            "@{}//{}",'.format(
                            repo_name,
                            target if target.find(":") != -1 else ":" + target,
                        ),
                    )
                else:
                    # Platform doesn't have this component for this version, use dummy.
                    build_content.append('            "{}",'.format(dummy_target))
            if platform_available and default_version and default_version in repos_dict:
                repo_name = repos_dict[default_version]
                build_content.append(
                    '        "//conditions:default": "@{}//{}",'.format(
                        repo_name,
                        target if target.find(":") != -1 else ":" + target,
                    ),
                )
            else:
                build_content.append('        "//conditions:default": ":unsupported_cuda_version",')

            build_content.append("    }),")
            build_content.append('    visibility = ["//visibility:public"],')
            build_content.append(")")
            build_content.append("")

    # Write the BUILD.bazel file
    ctx.file("BUILD.bazel", "\n".join(build_content))

platform_alias_repo = repository_rule(
    implementation = _platform_alias_repo_impl,
    attrs = dict({
        "component_name": attr.string(
            mandatory = True,
            doc = "Name of the component",
        ),
        "versions": attr.string_list(
            mandatory = True,
            doc = "List of versions to create aliases for",
        ),
    }, **_PLATFORM_REPO_ATTRS),
)
