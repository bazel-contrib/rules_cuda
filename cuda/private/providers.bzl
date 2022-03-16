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
]

ArchSpecInfo = provider(
    "",
    fields = {
        "stage1_arch": "A virtual architecture, str",
        "stage2_archs": "A list of virtual or gpu architecture, list of str",
    },
)

CudaArchsInfo = provider(
    """Provides a list of CUDA archs to compile for.

    To retain the flexiblity of NVCC, the extended notation is adopted, see
    https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation

    cuda_archs spec grammar as follows:

        ARCH_SPECS   ::= ARCH_SPEC [ ';' ARCH_SPECS ]
        ARCH_SPEC    ::= [ VIRTUAL_ARCH ':' ] GPU_ARCHS
        GPU_ARCHS    ::= GPU_ARCH [ ',' GPU_ARCHS ]
        GPU_ARCH     ::= ( 'sm_' | 'lto_' ) ARCH_NUMBER
                       | VIRTUAL_ARCH
        VIRTUAL_ARCH ::= ( 'compute_' | 'lto_' ) ARCH_NUMBER
        ARCH_NUMBER  ::= (a string in predefined cuda_archs list)

    E.g.:
        - compute_80:sm_80,sm_86
          Use compute_80 PTX, generate cubin with sm_80 and sm_86, no PTX embedded
        - compute_80:compute_80,sm_80,sm_86
          Use compute_80 PTX, generate cubin with sm_80 and sm_86, PTX embedded
        - compute_80:compute_80
          Embed compute_80 PTX, fully relay on ptxas
        - sm_80,sm_86
          Same as "compute_80:sm_80,sm_86", the arch with minimum integer value will be automatically populated.
        - sm_80;sm_86
          Two specs used.
        - compute_80
          Same as "compute_80:compute_80"

    Best Practices:
        - Library supports a full range of archs from xx to yy, you should embed the yy PTX
        - Library supports a sparse range of archs from xx to yy, you should embed the xx PTX

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
    }
)

CudaToolchainConfigInfo = provider(
    """""",
    fields = {
        "action_configs": "A list of action_configs.",
        "artifact_name_patterns": "A list of artifact_name_patterns.",
        "cuda_path": "cuda toolkit root path",
        "features": "A list of features.",
        "toolchain_identifier": "nvcc or clang",
    },
)
