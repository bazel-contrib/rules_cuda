"""Assemble a merged CUDA Toolkit directory from deliverable component repos."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//cuda/private:rules/merge_directory.bzl", "OsInfo", "merge_directory_action")

def _component_root(f):
    """Return the component root directory for a file inside a component repo/package."""
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
        return None
    return prefix + parts[0]

def _rebase_include(inc, comp_roots, out_path):
    """Rebase an include directory from a component root onto out_path."""
    for root in comp_roots:
        if inc == root:
            return out_path
        if inc.startswith(root + "/"):
            return out_path + inc[len(root):]
    return None

def _merged_ctk_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.label.name)

    inputs = []
    comp_roots = []
    seen_roots = {}
    includes = []
    system_includes = []
    has_cc_info = False

    for target in ctx.attr.deps:
        if CcInfo in target:
            has_cc_info = True
            cc_ctx = target[CcInfo].compilation_context
            target_files = cc_ctx.headers.to_list()
            target_roots = []
            for f in target_files:
                inputs.append(f)
                root = _component_root(f)
                if root:
                    if root not in target_roots:
                        target_roots.append(root)
                    if root not in seen_roots:
                        seen_roots[root] = True
                        comp_roots.append(root)
            for src_list, dst_list in [
                (cc_ctx.includes.to_list(), includes),
                (cc_ctx.system_includes.to_list(), system_includes),
                (cc_ctx.external_includes.to_list(), system_includes),
            ]:
                for inc in src_list:
                    rebased = _rebase_include(inc, target_roots, out_dir.path)
                    if rebased and rebased not in dst_list:
                        dst_list.append(rebased)
        else:
            default_info = target[DefaultInfo]
            target_files = default_info.files.to_list()
            executable = default_info.files_to_run.executable
            if executable and executable not in target_files:
                target_files.append(executable)
            for f in target_files:
                inputs.append(f)
                root = _component_root(f)
                if root and root not in seen_roots:
                    seen_roots[root] = True
                    comp_roots.append(root)

    merge_directory_action(
        ctx,
        srcs = comp_roots,
        dst = out_dir,
        inputs = inputs,
        exclude_files = ["LICENSE", "version.json"],
        is_windows = ctx.attr._exec_is_windows[OsInfo].is_windows,
        mnemonic = "MergedCtk",
        progress_message = "Assembling merged CUDA Toolkit %{output}",
    )

    compilation_context = cc_common.create_compilation_context(
        headers = depset([out_dir] if has_cc_info else []),
        includes = depset(includes),
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
        "deps": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Component targets (filegroups like //:nvcc_all_files, executables, or CcInfo header targets) to merge into a single CTK directory.",
        ),
        "_exec_is_windows": attr.label(
            default = Label("//cuda/private:is_windows"),
            cfg = "exec",
        ),
    },
    provides = [DefaultInfo, CcInfo],
)
