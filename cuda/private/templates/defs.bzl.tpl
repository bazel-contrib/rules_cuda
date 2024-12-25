def if_local_cuda_toolkit(if_true, if_false = []):
    is_local_ctk = %{is_local_ctk}
    if is_local_ctk:
        return if_true
    else:
        return if_false
