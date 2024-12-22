def if_local_cuda(if_true, if_false = []):
    is_local_cuda = %{is_local_cuda}
    if is_local_cuda:
        return if_true
    else:
        return if_false
