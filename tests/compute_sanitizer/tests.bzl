"""GPU-independent tests of the Compute Sanitizer launcher."""

def _launcher_test_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name + ".sh")
    launcher = ctx.executable.launcher
    ctx.actions.expand_template(
        template = ctx.file._script,
        output = executable,
        substitutions = {"%{launcher}": ctx.workspace_name + "/" + launcher.short_path},
        is_executable = True,
    )
    return [DefaultInfo(
        executable = executable,
        runfiles = ctx.runfiles(files = [launcher]).merge(ctx.attr.launcher[DefaultInfo].default_runfiles),
    )]

launcher_test = rule(
    implementation = _launcher_test_impl,
    test = True,
    attrs = {
        "launcher": attr.label(executable = True, cfg = "target", mandatory = True),
        "_script": attr.label(default = ":launcher_test.sh.tpl", allow_single_file = True),
    },
)
