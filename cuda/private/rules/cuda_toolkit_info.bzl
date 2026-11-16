load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def _impl(ctx):
    version_major, version_minor = ctx.attr.version.split(".")[:2]
    expanded_path = ctx.expand_location(ctx.attr.path, ctx.attr.path_data)
    root_files = []
    if not expanded_path and ctx.file.cuda_h:
        root_files = _assemble_root(ctx)
        expanded_path = paths.dirname(root_files[0].dirname)
    runtime_files = []
    runtime_depsets = []
    for target in ctx.attr.device_runtime_static_libs:
        if CcInfo in target:
            # Prebuilt cc_library/cc_import archives are linking inputs, not
            # DefaultInfo outputs. cudadevrt.lib is exposed as an interface library.
            for linker_input in target[CcInfo].linking_context.linker_inputs.to_list():
                for library in linker_input.libraries:
                    archive = library.static_library or library.pic_static_library or library.interface_library
                    if archive:
                        runtime_files.append(archive)
        else:
            runtime_depsets.append(target[DefaultInfo].files)
    device_runtime_static_libs = depset(runtime_files, transitive = runtime_depsets)

    return [DefaultInfo(files = depset(root_files)), CudaToolkitInfo(
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
        device_runtime_static_libs = device_runtime_static_libs,
    )]

def _assemble_root(ctx):
    # Clang locates ptxas, fatbinary and libdevice relative to `--cuda-path`
    # and detects the toolkit version from its include/cuda.h.
    root_files = []
    for directory, file in [
        ("bin", ctx.file.ptxas),
        ("bin", ctx.file.fatbinary),
        ("nvvm/libdevice", ctx.file.libdevice),
        ("include", ctx.file.cuda_h),
    ]:
        if not file:
            fail("cuda_h requires ptxas, fatbinary and libdevice to assemble a CUDA Toolkit root")
        link = ctx.actions.declare_file(paths.join(ctx.label.name, directory, file.basename))
        ctx.actions.symlink(output = link, target_file = file)
        root_files.append(link)
    return root_files

cuda_toolkit_info = rule(
    doc = """This rule provides CudaToolkitInfo.""",
    implementation = _impl,
    attrs = {
        "path": attr.string(mandatory = True, doc = "Root path to the CUDA Toolkit. Will expand location. If empty and `cuda_h` is set, " +
                                                    "the root is assembled from `ptxas`, `fatbinary`, `libdevice` and `cuda_h`, " +
                                                    "and its files are the default outputs of this target."),
        "path_data": attr.label_list(mandatory = False, doc = "Required if expand location."),
        "version": attr.string(mandatory = True, doc = "Version of the CUDA Toolkit."),
        "nvlink": attr.label(allow_single_file = True, cfg = "exec", doc = "The nvlink executable."),
        "link_stub": attr.label(allow_single_file = True, cfg = "exec", doc = "The link.stub text file."),
        "bin2c": attr.label(allow_single_file = True, cfg = "exec", doc = "The bin2c executable."),
        "fatbinary": attr.label(allow_single_file = True, cfg = "exec", doc = "The fatbinary executable."),
        "ptxas": attr.label(allow_single_file = True, cfg = "exec", doc = "The ptxas executable."),
        "cicc": attr.label(default = None, allow_single_file = True, cfg = "exec", doc = "The cicc executable."),
        "cuda_h": attr.label(default = None, allow_single_file = True, doc = "The cuda.h header of the CUDA Toolkit."),
        "libdevice": attr.label(default = None, allow_single_file = True, cfg = "exec", doc = "The libdevice LLVM bitcode library."),
        "device_runtime_static_libs": attr.label_list(allow_files = True, doc = "Static libraries needed for RDC device link and final host link."),
    },
    provides = [CudaToolkitInfo],
)
