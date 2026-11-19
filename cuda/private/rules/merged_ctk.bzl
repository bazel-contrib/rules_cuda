"""Assemble a merged CUDA Toolkit directory from deliverable component repos."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

_WINDOWS_COPY_BAT = """@echo off
setlocal
set "OUT=%~1"
shift
if not exist "%OUT%" mkdir "%OUT%"
:loop
if "%~1"=="" goto done
robocopy "%~1" "%OUT%" /E /XF LICENSE version.json /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 exit /b 1
shift
goto loop
:done
exit /b 0
"""

_POSIX_COPY_SH = """set -eu
out="$1"
shift
mkdir -p "$out"
for src in "$@"; do
    for entry in "$src"/*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        base="${entry##*/}"
        case "$base" in
            LICENSE|version.json) continue ;;
        esac
        cp -RLf "$entry" "$out/"
    done
    find "$out" -type d -exec chmod u+w {} +
done
"""

def _split_component_path(f):
    """Return (component_root, rel_path_within_component) for a component file."""
    path = f.path
    prefix = ""
    if f.root.path:
        prefix = f.root.path + "/"
    if f.owner.workspace_root:
        prefix += f.owner.workspace_root + "/"
    if f.owner.package:
        prefix += f.owner.package + "/"

    if prefix and path.startswith(prefix):
        rest = path[len(prefix):]
    else:
        rest = path

    parts = rest.split("/", 1)
    if len(parts) < 2:
        return None, None
    comp_dir, rel_path = parts[0], parts[1]
    return prefix + comp_dir, rel_path

def _merged_ctk_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.label.name)

    inputs = ctx.files.components
    comp_roots = []
    seen_roots = {}
    has_include = False
    has_cccl_include = False
    has_nvvm_include = False

    for f in inputs:
        comp_root, rel_path = _split_component_path(f)
        if not comp_root:
            continue
        if comp_root not in seen_roots:
            seen_roots[comp_root] = True
            comp_roots.append(comp_root)
        if rel_path.startswith("include/"):
            has_include = True
            if rel_path.startswith("include/cccl/"):
                has_cccl_include = True
        elif rel_path.startswith("nvvm/include/"):
            has_nvvm_include = True

    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )

    if is_windows:
        script = ctx.actions.declare_file(ctx.label.name + "_copy.bat")
        ctx.actions.write(
            output = script,
            content = _WINDOWS_COPY_BAT.replace("\n", "\r\n"),
            is_executable = True,
        )
        args = ctx.actions.args()
        args.add(out_dir.path.replace("/", "\\"))
        args.add_all([r.replace("/", "\\") for r in comp_roots])
        ctx.actions.run(
            executable = script,
            inputs = inputs,
            outputs = [out_dir],
            arguments = [args],
            mnemonic = "MergedCtk",
            progress_message = "Assembling merged CUDA Toolkit %{output}",
        )
    else:
        args = ctx.actions.args()
        args.add(out_dir.path)
        args.add_all(comp_roots)
        ctx.actions.run_shell(
            inputs = inputs,
            outputs = [out_dir],
            arguments = [args],
            mnemonic = "MergedCtk",
            progress_message = "Assembling merged CUDA Toolkit %{output}",
            command = _POSIX_COPY_SH,
        )

    system_includes = []
    if has_include:
        system_includes.append(out_dir.path + "/include")
    if has_cccl_include:
        system_includes.append(out_dir.path + "/include/cccl")
    if has_nvvm_include:
        system_includes.append(out_dir.path + "/nvvm/include")

    compilation_context = cc_common.create_compilation_context(
        headers = depset([out_dir]),
        system_includes = depset(system_includes),
    )

    return [
        DefaultInfo(files = depset([out_dir])),
        CcInfo(compilation_context = compilation_context),
    ]

merged_ctk = rule(
    doc = "Copies deliverable CUDA component trees into a single merged CTK directory and provides DefaultInfo and CcInfo.",
    implementation = _merged_ctk_impl,
    attrs = {
        "components": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Component filegroups (e.g. //:nvcc_all_files, //:cudart_all_files) to merge into a single CTK directory.",
        ),
        "_windows_constraint": attr.label(default = "@platforms//os:windows"),
    },
    provides = [DefaultInfo, CcInfo],
)
