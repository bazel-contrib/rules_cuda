"""Executable wrapper for running Bazel tests under Compute Sanitizer."""

def _runfiles_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

def _compute_sanitizer_launcher_impl(ctx):
    binaries = ctx.attr.sanitizer[DefaultInfo].files.to_list()
    if len(binaries) != 1:
        fail("Compute Sanitizer was not found. Install it in the local CUDA toolkit, or include the 'sanitizer' component in cuda.redist_json.")
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = executable,
        substitutions = {"%{sanitizer}": _runfiles_path(ctx, binaries[0])},
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = binaries)
    for dep in ctx.attr.data + [ctx.attr.sanitizer, ctx.attr._runfiles]:
        runfiles = runfiles.merge(ctx.runfiles(transitive_files = dep[DefaultInfo].files))
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = executable, runfiles = runfiles)]

compute_sanitizer_launcher = rule(
    implementation = _compute_sanitizer_launcher_impl,
    executable = True,
    attrs = {
        "sanitizer": attr.label(mandatory = True, allow_files = True),
        "data": attr.label_list(allow_files = True),
        "_template": attr.label(default = ":launcher.sh.tpl", allow_single_file = True),
        "_runfiles": attr.label(default = "@bazel_tools//tools/bash/runfiles"),
    },
)
