def if_cuda(if_true, if_false = []):
    return select({
        "@rules_cuda//cuda:is_enabled": if_true,
        "//conditions:default": if_false,
    })
