load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:defs.bzl", "cc_import")

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

    # NOTE: only empty when the component is not installed on the system, say, cublas is not installed with apt-get
    so_paths = native.glob([shared_library + "*"], allow_empty = True)

    cc_library(
        name = name,
        srcs = so_paths,
        target_compatible_with = ["@platforms//os:linux"],
    )

def soname_versioned_so(shared_library):
    """Returns the SONAME-named file of a versioned shared library, or None.

    If <shared_library> is path/to/foo.so and the package contains, e.g.,
    path/to/foo.so.12 and path/to/foo.so.12.8.90, this returns path/to/foo.so.12,
    i.e. the shortest versioned match, which is the file named after the SONAME
    that the dynamic linker looks up at runtime.

    Args:
        shared_library: Path of the unversioned .so file.

    Returns:
        The path of the SONAME-named file, or None if no versioned file exists.
    """
    so_paths = native.glob([shared_library + ".*"], allow_empty = True)
    if not so_paths:
        return None
    return sorted(so_paths, key = lambda p: (len(p), p))[0]

def cc_import_static_and_versioned_so(name, static_library, shared_library, **kwargs):
    """Creates a cc_import with a static library and the SONAME-named shared library.

    Bazel links the static library into fully statically linked targets and the
    shared library into dynamically linked ones (e.g. tests with the default
    --dynamic_mode). The shared library is resolved with soname_versioned_so() so
    that the runtime DT_NEEDED entry resolves within the runfiles.

    Args:
        name: Name of the cc_import.
        static_library: Path of the static library.
        shared_library: Path of the unversioned .so file.
        **kwargs: Additional arguments forwarded to cc_import.
    """
    cc_import(
        name = name,
        static_library = static_library,
        shared_library = soname_versioned_so(shared_library),
        **kwargs
    )
