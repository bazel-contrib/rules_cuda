def if_linux(if_true, if_false = []):
    return select({
        "@platforms//os:linux": if_true,
        "//conditions:default": if_false,
    })

def if_windows(if_true, if_false = []):
    return select({
        "@platforms//os:windows": if_true,
        "//conditions:default": if_false,
    })
