load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def _impl(ctx):
    version_major, version_minor = ctx.attr.version.split(".")[:2]
    return CudaToolkitInfo(
        path = ctx.attr.path,
        version_major = int(version_major),
        version_minor = int(version_minor),
        nvlink = ctx.file.nvlink,
        link_stub = ctx.file.link_stub,
        bin2c = ctx.file.bin2c,
        fatbinary = ctx.file.fatbinary,
    )

cuda_toolkit = rule(
    implementation = _impl,
    attrs = {
        "path": attr.string(mandatory = True),
        "version": attr.string(mandatory = True),
        "nvlink": attr.label(allow_single_file = True),
        "link_stub": attr.label(allow_single_file = True),
        "bin2c": attr.label(allow_single_file = True),
        "fatbinary": attr.label(allow_single_file = True),
    },
    provides = [CudaToolkitInfo],
)
