"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "local_cuda")

cuda_toolkit = tag_class(attrs = {
    "name": attr.string(doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
})

def _find_modules(module_ctx):
    root = None
    our_module = None
    for mod in module_ctx.modules:
        if mod.is_root:
            root = mod
        if mod.name == "rules_cuda":
            our_module = mod
    if root == None:
        root = our_module
    if our_module == None:
        fail("Unable to find rules_cuda module")

    return root, our_module

def _init(module_ctx):
    # Toolchain configuration is only allowed in the root module, or in rules_cuda.
    root, rules_cuda = _find_modules(module_ctx)
    toolchains = root.tags.local_toolchain or rules_cuda.tags.local_toolchain

    registrations = {}
    for toolchain in toolchains:
        if toolchain.name in registrations.keys():
            if toolchain.toolkit_path == registrations[toolchain.name]:
                # No problem to register a matching toolchain twice
                continue
            fail("Multiple conflicting toolchains declared for name {} ({} and {}".format(toolchain.name, toolchain.toolkit_path, registrations[toolchain.name]))
        else:
            registrations[toolchain.name] = toolchain.toolkit_path
    for name, toolkit_path in registrations.items():
        local_cuda(name = name, toolkit_path = toolkit_path)

toolchain = module_extension(
    implementation = _init,
    tag_classes = {"local_toolchain": cuda_toolkit},
)
