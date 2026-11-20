# Derived from bazel-skylib's copy_directory rule:
# https://github.com/bazelbuild/bazel-skylib/blob/main/rules/private/copy_directory_private.bzl

"""Implementation of merge_directory macro, rule, and action helper.

Merges multiple source directories into a single output TreeArtifact using Bash
(on Linux/macOS/non-Windows) or cmd.exe + robocopy (on Windows, no Bash needed).
"""

OsInfo = provider(
    doc = "Information about the target platform's OS.",
    fields = ["is_windows"],
)

is_windows = rule(
    implementation = lambda ctx: OsInfo(
        is_windows = ctx.target_platform_has_constraint(
            ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
        ),
    ),
    attrs = {
        "_windows_constraint": attr.label(default = "@platforms//os:windows"),
    },
)

def _src_path(src):
    return src if type(src) == "string" else src.path

def _merge_cmd(ctx, srcs, dst, inputs, exclude_files, mnemonic, progress_message):
    # Write the full command with paths embedded into a .bat file and invoke via
    # cmd.exe /C to avoid MSVC-vs-cmd.exe argument quoting mismatches (same pattern
    # as bazel-skylib's copy_directory).
    bat = ctx.actions.declare_file("%s-%s-cmd.bat" % (ctx.label.name, hash(dst.path)))

    dst_win = dst.path.replace("/", "\\")
    xf_clause = ""
    if exclude_files:
        xf_clause = " /XF " + " ".join(exclude_files)

    lines = [
        "@echo off",
        "setlocal",
        'set "ROBOCOPY=C:\\Windows\\System32\\robocopy.exe"',
        'if defined SystemRoot set "ROBOCOPY=%SystemRoot%\\System32\\robocopy.exe"',
        'if not exist "{dst}" mkdir "{dst}"'.format(dst = dst_win),
    ]
    for src in srcs:
        src_win = _src_path(src).replace("/", "\\")
        lines.extend([
            'if not exist "{src}\\" ('.format(src = src_win),
            '  echo Error: "{src}" is not a directory'.format(src = src_win),
            "  exit /b 1",
            ")",
            # robocopy returns 0..7 on success and >=8 on errors.
            '"%ROBOCOPY%" "{src}" "{dst}" /E{xf} /NFL /NDL /NJH /NJS /NC /NS /NP >nul'.format(
                src = src_win,
                dst = dst_win,
                xf = xf_clause,
            ),
            "if errorlevel 8 exit /b 1",
        ])
    lines.append("exit /b 0")

    ctx.actions.write(
        output = bat,
        content = "\r\n".join(lines) + "\r\n",
        is_executable = True,
    )

    action_inputs = depset(direct = [bat], transitive = [inputs]) if type(inputs) == "depset" else list(inputs) + [bat]
    ctx.actions.run(
        inputs = action_inputs,
        outputs = [dst],
        executable = "cmd.exe",
        arguments = ["/C", bat.path.replace("/", "\\")],
        mnemonic = mnemonic,
        progress_message = progress_message,
        use_default_shell_env = True,
    )

def _merge_bash(ctx, srcs, dst, inputs, exclude_files, mnemonic, progress_message):
    exclude_block = ""
    if exclude_files:
        pattern = "|".join(exclude_files)
        exclude_block = """
        base="${entry##*/}"
        case "$base" in
            """ + pattern + """) continue ;;
        esac"""

    cmd = """set -eu
out="$1"
shift
mkdir -p "$out"
for src in "$@"; do
    if [ ! -d "$src" ]; then
        echo "Error: $src is not a directory" >&2
        exit 1
    fi
    for entry in "$src"/* "$src"/.[!.]* "$src"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue""" + exclude_block + """
        cp -RLf "$entry" "$out/"
    done
    find "$out" -type d -exec chmod u+w {} +
done
"""

    args = ctx.actions.args()
    args.add(dst.path)
    args.add_all([_src_path(s) for s in srcs])

    ctx.actions.run_shell(
        inputs = inputs,
        outputs = [dst],
        arguments = [args],
        command = cmd,
        mnemonic = mnemonic,
        progress_message = progress_message,
    )

def merge_directory_action(
        ctx,
        srcs,
        dst,
        inputs = None,
        exclude_files = [],
        is_windows = False,
        mnemonic = "MergeDirectory",
        progress_message = None):
    """Helper function that creates an action to merge directories from srcs into dst.

    Args:
        ctx: The rule context.
        srcs: List of source directory paths (strings) or directory Files to merge in order.
        dst: The output directory TreeArtifact.
        inputs: List or depset of input Files for the action. Defaults to srcs when srcs are Files.
        exclude_files: Optional list of top-level file basenames to skip when merging.
        is_windows: If true, a cmd.exe + robocopy action is created with no bash dependency.
        mnemonic: Action mnemonic (default "MergeDirectory").
        progress_message: Action progress message (default "Merging directories into %{output}").
    """
    if dst.is_source or not dst.is_directory:
        fail("dst must be a TreeArtifact")
    if inputs == None:
        inputs = srcs
    if progress_message == None:
        progress_message = "Merging directories into %{output}"

    if is_windows:
        _merge_cmd(ctx, srcs, dst, inputs, exclude_files, mnemonic, progress_message)
    else:
        _merge_bash(ctx, srcs, dst, inputs, exclude_files, mnemonic, progress_message)

def _merge_directory_impl(ctx):
    dst = ctx.actions.declare_directory(ctx.attr.out)
    merge_directory_action(
        ctx,
        srcs = ctx.files.srcs,
        dst = dst,
        exclude_files = ctx.attr.exclude_files,
        is_windows = ctx.attr._exec_is_windows[OsInfo].is_windows,
    )
    files = depset(direct = [dst])
    runfiles = ctx.runfiles(files = [dst])
    return [DefaultInfo(files = files, runfiles = runfiles)]

_merge_directory = rule(
    implementation = _merge_directory_impl,
    provides = [DefaultInfo],
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Directories (source directories or TreeArtifacts) to merge in order.",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Path of the output directory, relative to this package.",
        ),
        "exclude_files": attr.string_list(
            default = [],
            doc = "Optional list of top-level file basenames to skip when merging.",
        ),
        "_exec_is_windows": attr.label(
            default = Label("//cuda/private:is_windows"),
            cfg = "exec",
        ),
    },
)

def merge_directory(name, srcs, out, exclude_files = [], **kwargs):
    """Merges multiple directories into a single output directory TreeArtifact.

    Args:
        name: Name of the rule.
        srcs: List of source directories or TreeArtifacts to merge in order.
        out: Path of the output directory, relative to this package.
        exclude_files: Optional list of top-level file basenames to skip when merging.
        **kwargs: Further keyword arguments, e.g. `visibility`.
    """
    _merge_directory(
        name = name,
        srcs = srcs,
        out = out,
        exclude_files = exclude_files,
        **kwargs
    )
