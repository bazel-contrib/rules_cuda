"""Defines all providers that are used in this repo."""

cuda_archs = [
    "30",
    "32",
    "35",
    "37",
    "50",
    "52",
    "53",
    "60",
    "61",
    "62",
    "70",
    "72",
    "75",
    "80",
    "86",
    "87",
    "89",
    "90",
    "90a",
    "100",
    "100a",
    "100f",
    "101",
    "101a",
    "101f",
    "103",
    "103a",
    "103f",
    "120",
    "120a",
    "120f",
    "121",
    "121a",
    "121f",
]

Stage2ArchInfo = provider(
    """Provides the information of how the stage 2 compilation is carried out.

One and only one of `virtual`, `gpu` and `lto` must be set to True. For example, if `arch` is set to `80` and `virtual` is `True`, then a
ptx embedding process is carried out for `compute_80`. Multiple `Stage2ArchInfo` can be used for specifying how a stage 1 result is
transformed into its final form.""",
    fields = {
        "arch": "str, arch number",
        "virtual": "bool, use virtual arch, default False",
        "gpu": "bool, use gpu arch, default False",
        "lto": "bool, use lto, default False",
    },
)

ArchSpecInfo = provider(
    """Provides the information of how [GPU compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation)
is carried out of a single virtual architecture.""",
    fields = {
        "stage1_arch": "A virtual architecture, str, arch number only",
        "stage2_archs": "A list of virtual or gpu architecture, list of Stage2ArchInfo",
    },
)

CudaArchsInfo = provider(
    """Provides a list of CUDA archs to compile for.

Read the whole [Chapter 5 of CUDA Compiler Driver NVCC Reference Guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation)
if more detail is needed.""",
    fields = {
        "arch_specs": "A list of ArchSpecInfo",
    },
)

CudaInfo = provider(
    """Provides cuda build artifacts that can be consumed by device linking or linking process.

This provider is analog to [CcInfo](https://bazel.build/rules/lib/CcInfo) but only contains necessary information for
linking in a flat structure. Objects are grouped by direct and transitive, because we have no way to split them again
if merged a single depset.
""",
    fields = {
        "defines": "A depset of strings. It is used for the compilation during device linking.",
        # direct only:
        "objects": "A depset of objects. Direct artifacts of the rule.",  # but not rdc and pic
        "pic_objects": "A depset of position independent code objects. Direct artifacts of the rule.",  # but not rdc
        "rdc_objects": "A depset of relocatable device code objects. Direct artifacts of the rule.",  # but not pic
        "rdc_pic_objects": "A depset of relocatable device code and position independent code objects. Direct artifacts of the rule.",
        # transitive archive only (cuda_objects):
        "archive_objects": "A depset of rdc objects. cuda_objects only. Gathered from the transitive dependencies for archiving.",
        "archive_pic_objects": "A depset of rdc pic objects. cuda_objects only. Gathered from the transitive dependencies for archiving.",
        "archive_rdc_objects": "A depset of rdc objects. cuda_objects only. Gathered from the transitive dependencies for archiving or device linking.",
        "archive_rdc_pic_objects": "A depset of rdc pic objects. cuda_objects only. Gathered from the transitive dependencies for archiving or device linking.",

        # transitive dlink only (cuda_library):
        # NOTE: ideally, we can use the archived library to do the device linking, but the nvlink is not happy with library with *_dlink.o included
        "dlink_rdc_objects": "A depset of rdc objects. cuda_library only. Gathered from the transitive dependencies for device linking.",
        "dlink_rdc_pic_objects": "A depset of rdc pic objects. cuda_library only. Gathered from the transitive dependencies for device linking.",
    },
)

CudaToolkitInfo = provider(
    """Provides the information of CUDA Toolkit.""",
    fields = {
        "path": "string of path to cuda toolkit root",
        "version_major": "int of the cuda toolkit major version, e.g, 11 for 11.6",
        "version_minor": "int of the cuda toolkit minor version, e.g, 6 for 11.6",
        "nvlink": "File to the nvlink executable",
        "link_stub": "File to the link.stub file",
        "bin2c": "File to the bin2c executable",
        "fatbinary": "File to the fatbinary executable",
    },
)

CudaToolchainConfigInfo = provider(
    """Provides the information of what the toolchain is and how the toolchain is configured.""",
    fields = {
        "action_configs": "A list of action_configs.",
        "artifact_name_patterns": "A list of artifact_name_patterns.",
        "cuda_toolkit": "A target that provides a `CudaToolkitInfo`",
        "features": "A list of features.",
        "toolchain_identifier": "nvcc or clang",
    },
)
