load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def _impl(ctx):
    version_major, version_minor = ctx.attr.version.split(".")[:2]
    expanded_path = ctx.expand_location(ctx.attr.path, ctx.attr.path_data)

    return CudaToolkitInfo(
        path = expanded_path,
        version_major = int(version_major),
        version_minor = int(version_minor),
        nvlink = ctx.file.nvlink,
        link_stub = ctx.file.link_stub,
        bin2c = ctx.file.bin2c,
        fatbinary = ctx.file.fatbinary,
        ptxas = ctx.file.ptxas,
        cicc = ctx.file.cicc,
        libdevice = ctx.file.libdevice,
    )

cuda_toolkit_info = rule(
    doc = """This rule provides CudaToolkitInfo.""",
    implementation = _impl,
    attrs = {
        "path": attr.string(mandatory = True, doc = "Root path to the CUDA Toolkit. Will expand location."),
        "path_data": attr.label_list(mandatory = False, doc = "Required if expand location."),
        "version": attr.string(mandatory = True, doc = "Version of the CUDA Toolkit."),
        "nvlink": attr.label(allow_single_file = True, doc = "The nvlink executable."),
        "link_stub": attr.label(allow_single_file = True, doc = "The link.stub text file."),
        "bin2c": attr.label(allow_single_file = True, doc = "The bin2c executable."),
        "fatbinary": attr.label(allow_single_file = True, doc = "The fatbinary executable."),
        "ptxas": attr.label(allow_single_file = True, doc = "The ptxas executable."),
        "cicc": attr.label(default = None, allow_single_file = True, doc = "The cicc executable."),
        "libdevice": attr.label(default = None, allow_single_file = True, doc = "The libdevice LLVM bitcode library."),
    },
    provides = [CudaToolkitInfo],
)
