"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "local_cuda")

cuda_toolkit = tag_class(attrs = {
    "name": attr.string(doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
})

def _init(module_ctx):
    registrations = {}
    # first pass only handles root module
    for mod in module_ctx.modules:
        if not mod.is_root:
            continue
        for toolchain in mod.tags.local_toolchain:
            if toolchain.name in registrations.keys():
                fail("Multiple toolchains defined with the name \"{}\" during initialization of module \"{}\"".format(toolchain.name, mod.name))
            else:
                registrations[toolchain.name] = toolchain.toolkit_path
    # second pass only handles non-root module
    for mod in module_ctx.modules:
        if mod.is_root:
            continue
        for toolchain in mod.tags.local_toolchain:
            # root module supersede non-root modules
            if toolchain.name not in registrations.keys():
                registrations[toolchain.name] = toolchain.toolkit_path
    for name, toolkit_path in registrations.items():
        local_cuda(name = name, toolkit_path = toolkit_path)

toolchain = module_extension(
    implementation = _init,
    tag_classes = {"local_toolchain": cuda_toolkit},
)
