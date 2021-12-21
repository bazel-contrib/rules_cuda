def detect_cuda_toolkit():
    print("dummy implementation")
    native.register_toolchains(
        "//cuda/private:nvcc-ubuntu-toolchain"
    )
