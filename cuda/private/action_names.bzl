"""Constants for action names used for Cuda rules."""

# cuda compile comprise of host and device compilation
CUDA_COMPILE = "cuda-compile"

CUDA_DEVICE_LINK = "cuda-dlink"

ACTION_NAMES = struct(
    cuda_compile = CUDA_COMPILE,
    device_link = CUDA_DEVICE_LINK,
)
