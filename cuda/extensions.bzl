"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "local_cuda")

cuda_toolkit = tag_class(attrs = {
    "name": attr.string(doc = "Name for the toolchain repository"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK"),
})

def _init(module_ctx):
    registrations = {}
    for mod in module_ctx.modules:
        for toolchain in mod.tags.toolchain:
            if toolchain.name in registrations.keys():
                if toolchain.toolkit_path == registrations[toolchain.name]:
                    # No problem to register a matching toolchain twice
                    continue
                fail("Multiple conflicting toolchains declared for name {} ({} and {}".format(toolchain.name, toolchain.toolkit_path, registrations[toolchain.name]))
            else:
                registrations[toolchain.name] = toolchain.toolkit_path
    for name, toolkit_path in registrations.items():
        local_cuda(name = name, toolkit_path = toolkit_path)

toolchain = module_extension(implementation = _init, tag_classes = {"toolchain": cuda_toolkit})
