load("//cuda/private:providers.bzl", _CudaArchsInfo = "CudaArchsInfo", _cuda_archs = "cuda_archs")
load("//cuda/private:rules/cuda_objects.bzl", _cuda_objects = "cuda_objects")
load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")
load(
    "//cuda/private:toolchain.bzl",
    _cuda_toolchain = "cuda_toolchain",
    _find_cuda_toolchain = "find_cuda_toolchain",
    _use_cuda_toolchain = "use_cuda_toolchain",
)
load("//cuda/private/toolchain_configs/windows:toolchain_config.bzl", _cuda_toolchain_config_windows = "cuda_toolchain_config")
load("//cuda/private/toolchain_configs/linux:toolchain_config.bzl", _cuda_toolchain_config_linux = "cuda_toolchain_config")

cuda_toolchain = _cuda_toolchain
find_cuda_toolchain = _find_cuda_toolchain
use_cuda_toolchain = _use_cuda_toolchain
cuda_toolchain_config_windows = _cuda_toolchain_config_windows
cuda_toolchain_config_linux = _cuda_toolchain_config_linux

cuda_archs = _cuda_archs
CudaArchsInfo = _CudaArchsInfo

cuda_objects = _cuda_objects
cuda_library = _cuda_library
