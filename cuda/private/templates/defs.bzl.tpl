def if_local_cuda_toolkit(if_true, if_false = []):
    is_local_ctk = %{is_local_ctk}
    if is_local_ctk:
        return if_true
    else:
        return if_false

def if_deliverable_cuda_toolkit(if_true, if_false = []):
    return if_local_cuda_toolkit(if_false, if_true)

def if_cuda_clang(if_true, if_false = []):
    return select({
        "@rules_cuda//cuda:compiler_is_clang": if_true,
        "//conditions:default": if_false,
    })

def additional_header_deps(component_name):
    if component_name == "cudart":
        return if_deliverable_cuda_toolkit([
            "@cuda//:nvcc_headers",
            "@cuda//:cccl_headers",
        ]) + if_cuda_clang(if_deliverable_cuda_toolkit([
            "@cuda//:curand_headers",
        ]))

    return []
