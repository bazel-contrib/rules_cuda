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
    "90"
]

Stage2ArchInfo = provider(
    "",
    fields = {
        "arch": "str, arch number",
        "virtual": "bool, use virtual arch, default False",
        "gpu": "bool, use gpu arch, default False",
        "lto": "bool, use lto, default False",
    },
)

ArchSpecInfo = provider(
    "",
    fields = {
        "stage1_arch": "A virtual architecture, str, arch number only",
        "stage2_archs": "A list of virtual or gpu architecture, list of Stage2ArchInfo",
    },
)

CudaArchsInfo = provider(
    """Provides a list of CUDA archs to compile for.

    Read the whole chapter 5 of CUDA Compiler Driver NVCC Reference Guide, at
    https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation
    """,
    fields = {
        "arch_specs": "A list of ArchSpecInfo",
    },
)

CudaInfo = provider(
    """Provider that wraps cuda build information.""",
    fields = {
        "defines": "A depset of strings",
        "objects": "A depset of objects.",  # but not rdc and pic
        "rdc_objects": "A depset of relocatable device code objects.",  # but not pic
        "pic_objects": "A depset of position indepentent code objects.",  # but not rdc
        "rdc_pic_objects": "A depset of relocatable device code and position indepentent code objects.",
    },
)

CudaToolkitInfo = provider(
    "",
    fields = {
        "path": "string of path to cuda toolkit root",
        "version_major": "int of the cuda toolkit major version, e.g, 11 for 11.6",
        "version_minor": "int of the cuda toolkit minor version, e.g, 6 for 11.6",
        "nvlink": "File to the nvlink executable",
        "link_stub": "File to the link.stub file",
        "bin2c": "File to the bin2c executable",
        "fatbinary": "File to the fatbinary executable",
    }
)

CudaToolchainConfigInfo = provider(
    """""",
    fields = {
        "action_configs": "A list of action_configs.",
        "artifact_name_patterns": "A list of artifact_name_patterns.",
        "cuda_toolkit": "CudaToolkitInfo",
        "features": "A list of features.",
        "toolchain_identifier": "nvcc or clang",
    },
)
