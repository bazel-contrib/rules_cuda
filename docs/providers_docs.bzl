load(
    "@rules_cuda//cuda/private:providers.bzl",
    _ArchSpecInfo = "ArchSpecInfo",
    _CudaArchsInfo = "CudaArchsInfo",
    _CudaInfo = "CudaInfo",
    _CudaToolchainConfigInfo = "CudaToolchainConfigInfo",
    _CudaToolkitInfo = "CudaToolkitInfo",
    _Stage2ArchInfo = "Stage2ArchInfo",
)

ArchSpecInfo = _ArchSpecInfo
Stage2ArchInfo = _Stage2ArchInfo

CudaArchsInfo = _CudaArchsInfo
CudaInfo = _CudaInfo
CudaToolkitInfo = _CudaToolkitInfo
CudaToolchainConfigInfo = _CudaToolchainConfigInfo
