load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")

def glob(query_string):
    return native.glob(include = [query_string])

def cuda_library_with_per_object_optf(name = None, srcs = [], deps = [], **kwargs):
    if name == None:
        fail("Library target name required.")
    if "optf" in kwargs:
        fail("If you want a global optf file, use cuda_library() instead.")

    objects = []
    optfs = glob("kernels/*.optf")
    for kernel in srcs:
        optf_name = kernel + ".optf"
        object_name = kernel + "_obj"
        objects.append(object_name)
        optf = None
        if optf_name in optfs:
            print("optf found for kernel", kernel)
            optf = optf_name
        else:
            print("optf NOT found for kernel", kernel)
        cuda_objects(
            name = object_name,
            srcs = [kernel],
            optf = optf,
            deps = deps,
            **kwargs
        )

    library_deps = objects + deps
    cuda_library(
        name = name,
        deps = library_deps,
        rdc = True,
        **kwargs
    )
