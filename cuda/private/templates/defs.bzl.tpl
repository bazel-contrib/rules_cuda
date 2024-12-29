def if_local_cuda_toolkit(if_true, if_false = []):
    is_local_ctk = %{is_local_ctk}
    if is_local_ctk:
        return if_true
    else:
        return if_false

def additional_header_deps(component_name):
    if component_name == "cudart":
        return if_local_cuda_toolkit([
            "@local_cuda//:nvcc_headers",
            "@local_cuda//:cccl_headers",
        ])

    return []
