def register_rule_dev_toolchains(use_detected = False):
    if use_detected:
        register_detected_cuda_toolchains()
    else:
        native.register_toolchains(
            "//cuda/private/toolchain_configs/linux:nvcc-ubuntu-toolchain",
            "//cuda/private/toolchain_configs/windows:nvcc-windows-toolchain",
        )

def register_detected_cuda_toolchains():
    native.register_toolchains(
        "@local_cuda//toolchain:nvcc-local-toolchain",
    )
