# map short component name to consumable targets
REGISTRY = {
    "cudart": ["cudart_all_files", "cudart_license", "cudart_headers", "cuda", "cuda_runtime", "cuda_runtime_static", "cudadevrt_a", "cudadevrt_lib", "culibos_a"],  # device-runtime targets are not for end users.
    "nvcc": ["nvcc_all_files", "nvcc_license", "nvcc_headers", "compiler_root", "compiler_deps", "nvptxcompiler", "nvcc", "nvlink", "ptxas", "bin2c", "fatbinary", "link.stub", "cicc", "libdevice", "libdevice.10.bc"],
    "nvvm": ["nvvm_all_files", "nvvm_license", "nvvm_headers", "cicc", "libdevice", "libdevice.10.bc"],
    "cccl": ["cccl_all_files", "cccl_license", "cccl_headers", "libcudacxx", "cub", "thrust"],
    "crt": ["crt_all_files", "crt_license", "crt_headers", "crt"],
    "culibos": ["culibos_all_files", "culibos_license", "culibos_headers", "culibos_a"],  # culibos_a is not for end users.
    "cublas": ["cublas_all_files", "cublas_license", "cublas_headers", "cublas"],
    "cufft": ["cufft_all_files", "cufft_license", "cufft_headers", "cufft", "cufft_static"],
    "cufile": ["cufile", "cufile_all_files", "cufile_header_files", "cufile_headers", "cufile_license"],
    "cupti": ["cupti_all_files", "cupti_license", "cupti_headers", "cupti", "nvperf_host", "nvperf_target"],
    "curand": ["curand_all_files", "curand_license", "curand_headers", "curand"],
    "cusolver": ["cusolver_all_files", "cusolver_license", "cusolver_headers", "cusolver"],
    "cusparse": ["cusparse_all_files", "cusparse_license", "cusparse_headers", "cusparse"],
    "npp": ["npp_all_files", "npp_license", "npp_headers", "npp", "nppc", "nppi", "nppial", "nppicc", "nppidei", "nppif", "nppig", "nppim", "nppist", "nppisu", "nppitc", "npps"],
    "nvidia_fs": [],
    "nvjitlink": ["nvjitlink_all_files", "nvjitlink_license", "nvjitlink_headers", "nvjitlink", "nvjitlink_static"],
    "nvjpeg": ["nvjpeg_all_files", "nvjpeg_license", "nvjpeg_headers", "nvjpeg", "nvjpeg_static"],
    "nvml": ["nvml_all_files", "nvml_license", "nvml_headers", "nvml"],
    "nvprof": [],
    "nvrtc": ["nvrtc_all_files", "nvrtc_license", "nvrtc_headers", "nvrtc"],
    "nvtx": ["nvtx_all_files", "nvtx_license", "nvtx_headers", "nvtx", "nvtx3"],
}

# Map short component names to manifest names by CUDA Toolkit version.
# Version keys are exclusive upper bounds; "latest" covers the final interval.
# Availability is determined by the manifest, not by this name mapping.
# Audited against the latest patch of each published CUDA 11.0-13.5 series;
# see tests/redist_json/manifest_component_names.bzl for the source snapshots.
FULL_COMPONENT_NAME = {
    "cudart": {"latest": "cuda_cudart"},
    "nvcc": {"latest": "cuda_nvcc"},
    "nvvm": {"latest": "libnvvm"},
    "cccl": {
        "13.3.0": "cuda_cccl",
        "latest": "cccl",
    },
    "crt": {"latest": "cuda_crt"},
    "culibos": {"latest": "cuda_culibos"},
    "cublas": {"latest": "libcublas"},
    "cufft": {"latest": "libcufft"},
    "cufile": {"latest": "libcufile"},
    "cupti": {"latest": "cuda_cupti"},
    "curand": {"latest": "libcurand"},
    "cusolver": {"latest": "libcusolver"},
    "cusparse": {"latest": "libcusparse"},
    "npp": {"latest": "libnpp"},
    "nvidia_fs": {"latest": "nvidia_fs"},
    "nvjitlink": {"latest": "libnvjitlink"},
    "nvjpeg": {"latest": "libnvjpeg"},
    "nvml": {"latest": "cuda_nvml_dev"},
    "nvrtc": {"latest": "cuda_nvrtc"},
    "nvtx": {"latest": "cuda_nvtx"},
}

def _version_tuple(version):
    parts = version.split(".")
    if len(parts) < 2 or len(parts) > 3 or not all([p.isdigit() for p in parts]):
        fail("Expected CUDA Toolkit version major.minor[.patch], got {}".format(repr(version)))
    return tuple([int(p) for p in parts] + [0] * (3 - len(parts)))

def get_full_component_name(component, version):
    """Return the manifest key for a component at a CUDA Toolkit version.

    Args:
        component: Short component name, such as "cccl".
        version: CUDA Toolkit major.minor[.patch], or "latest".

    Returns:
        The full component name used in that toolkit release's redist manifest.
    """
    names = FULL_COMPONENT_NAME[component]
    if version == "latest":
        return names["latest"]
    requested_version = _version_tuple(version)
    for upper_bound in sorted([v for v in names if v != "latest"], key = _version_tuple):
        if requested_version < _version_tuple(upper_bound):
            return names[upper_bound]
    return names["latest"]
