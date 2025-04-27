"""cuda_toolkit_info implementation"""

load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def _cuda_toolkit_info_impl(ctx):
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

cuda_toolkit_info = rule(
    doc = """This rule provides CudaToolkitInfo.""",
    implementation = _cuda_toolkit_info_impl,
    attrs = {
        "bin2c": attr.label(
            doc = "The bin2c executable.",
            allow_single_file = True,
        ),
        "fatbinary": attr.label(
            doc = "The fatbinary executable.",
            allow_single_file = True,
        ),
        "link_stub": attr.label(
            doc = "The link.stub text file.",
            allow_single_file = True,
        ),
        "nvlink": attr.label(
            doc = "The nvlink executable.",
            allow_single_file = True,
        ),
        "path": attr.string(
            doc = "Root path to the CUDA Toolkit.",
            mandatory = True,
        ),
        "version": attr.string(
            doc = "Version of the CUDA Toolkit.",
            mandatory = True,
        ),
    },
    provides = [CudaToolkitInfo],
)
