def register_detected_cuda_toolchains():
    native.register_toolchains(
        "@local_cuda//toolchain:nvcc-local-toolchain",
        "@local_cuda//toolchain/clang:clang-local-toolchain",
    )
