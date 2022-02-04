load("//cuda/private:providers.bzl", _cuda_archs = "cuda_archs", _CudaArchsInfo = "CudaArchsInfo")
load("//cuda/private:rules/cuda_objects.bzl", _cuda_objects = "cuda_objects")
load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

cuda_archs = _cuda_archs
CudaArchsInfo = _CudaArchsInfo

cuda_objects = _cuda_objects
cuda_library = _cuda_library
