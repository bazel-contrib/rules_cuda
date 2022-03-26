def detect_cuda_toolkit():
    print("dummy implementation")
    native.register_toolchains(
        "//cuda/private/toolchain_configs/linux:nvcc-ubuntu-toolchain",
        "//cuda/private/toolchain_configs/windows:nvcc-windows-toolchain"
    )
