def _cuda_toolchain_impl(ctx):
    print(ctx.label.name)
    return [
     platform_common.ToolchainInfo(
        name = ctx.label.name,
        compiler_driver = ctx.attr.compiler_driver,
        compiler_executable = ctx.attr.compiler_executable,
        cuda_archs = ctx.attr.cuda_archs,
        include_directories = ctx.attr.include_directories,
        lib_directories = ctx.attr.lib_directories,
        bin_directories = ctx.attr.bin_directories,
        actions = struct(
            # compile = compile,
            # device_link = device_link,
        )
     )
    ]

cuda_toolchain = rule(
    implementation = _cuda_toolchain_impl,
    attrs = {
        "compiler_driver": attr.string(
            mandatory = True,
        ),
        "compiler_executable": attr.string(
            mandatory = True,
        ),
        "cuda_archs": attr.string_list(
            mandatory = True,
        ),
        "include_directories": attr.string_list(
            mandatory = True,
        ),
        "lib_directories": attr.string_list(
            mandatory = True,
        ),
        "bin_directories": attr.string_list(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
)

def find_cuda_toolchain(ctx):
    return ctx.toolchains["//cuda:toolchain_type"]
