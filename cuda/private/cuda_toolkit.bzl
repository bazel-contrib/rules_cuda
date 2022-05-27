def register_detected_cuda_toolchains():
    native.register_toolchains(
        "@local_cuda//toolchain:nvcc-local-toolchain",
    )
