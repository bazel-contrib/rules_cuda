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

def cc_import_versioned_sos(name, shared_library):
    """Creates a cc_library that depends on all versioned .so files with the given prefix.

    If <shared_library> is path/to/foo.so, and it is a symlink to foo.so.<version>,
    this should be used instead of cc_import.
    The versioned files are typically needed at runtime, but not at build time.

    Args:
        name: Name of the cc_library.
        shared_library: Prefix of the versioned .so files.
    """

    # NOTE: only empty when the componnent is not installed on the system, say, cublas is not installed with apt-get
    so_paths = native.glob([shared_library + "*"], allow_empty = True)

    native.cc_library(
        name = name,
        srcs = so_paths,
        target_compatible_with = ["@platforms//os:linux"],
    )
