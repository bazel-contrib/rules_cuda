"""Common component definitions."""

# map short component name to consumable targets
REGISTRY = {
    "cccl": ["cub", "thrust", "cccl_headers"],
    "cublas": ["cublas"],
    "cudart": ["cuda", "cuda_runtime", "cuda_runtime_static"],
    "cufft": ["cufft", "cufft_static"],
    "cufile": [],
    "cupti": ["cupti", "nvperf_host", "nvperf_target"],
    "curand": ["curand"],
    "cusolver": ["cusolver"],
    "cusparse": ["cusparse"],
    "npp": ["nppc", "nppi", "nppial", "nppicc", "nppidei", "nppif", "nppig", "nppim", "nppist", "nppisu", "nppitc", "npps"],
    "nvcc": ["compiler_deps", "nvptxcompiler", "nvcc_headers"],
    "nvidia_fs": [],
    "nvjitlink": ["nvjitlink", "nvjitlink_static"],
    "nvjpeg": ["nvjpeg", "nvjpeg_static"],
    "nvml": ["nvml"],
    "nvprof": [],
    "nvrtc": ["nvrtc"],
    "nvtx": ["nvtx"],
}

# map short component name to full component name
FULL_COMPONENT_NAME = {
    "cccl": "cuda_cccl",
    "cublas": "libcublas",
    "cudart": "cuda_cudart",
    "cufft": "libcufft",
    "cufile": "libcufile",
    "cupti": "libcupti",
    "curand": "libcurand",
    "cusolver": "libcusolver",
    "cusparse": "libcusparse",
    "npp": "libnpp",
    "nvcc": "cuda_nvcc",
    "nvidia_fs": "nvidia_fs",
    "nvjitlink": "libnvjitlink",
    "nvjpeg": "libnvjpeg",
    "nvml": "cuda_nvml_dev",
    "nvrtc": "cuda_nvrtc",
    "nvtx": "cuda_nvtx",
}
