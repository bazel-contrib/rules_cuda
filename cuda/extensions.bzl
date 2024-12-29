"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "local_cuda")

cuda_toolkit_tag = tag_class(attrs = {
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

def _module_tag_to_dict(t):
    return {attr: getattr(t, attr) for attr in dir(t)}

def _init(module_ctx):
    # Toolchain configuration is only allowed in the root module, or in rules_cuda.
    root, rules_cuda = _find_modules(module_ctx)
    toolkits = root.tags.toolkit or rules_cuda.tags.toolkit

    registrations = {}
    for toolkit in toolkits:
        if toolkit.name in registrations.keys():
            if toolkit.toolkit_path == registrations[toolkit.name].toolkit_path:
                # No problem to register a matching toolkit twice
                continue
            fail("Multiple conflicting toolkits declared for name {} ({} and {}".format(toolkit.name, toolkit.toolkit_path, registrations[toolkit.name].toolkit_path))
        else:
            registrations[toolkit.name] = toolkit
    for _, toolkit in registrations.items():
        local_cuda(**_module_tag_to_dict(toolkit))

toolchain = module_extension(
    implementation = _init,
    tag_classes = {"toolkit": cuda_toolkit_tag},
)
