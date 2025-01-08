"""Entry point for extensions used by bzlmod."""

load("//cuda/private:compat.bzl", "components_mapping_compat")
load("//cuda/private:repositories.bzl", "cuda_component", "local_cuda")

cuda_component_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True, doc = "Repo name for the deliverable cuda_component"),
    "component_name": attr.string(doc = "Short name of the component defined in registry."),
    "integrity": attr.string(
        doc = "Expected checksum in Subresource Integrity format of the file downloaded. " +
              "This must match the checksum of the file downloaded.",
    ),
    "sha256": attr.string(
        doc = "The expected SHA-256 of the file downloaded. " +
              "This must match the SHA-256 of the file downloaded.",
    ),
    "strip_prefix": attr.string(
        doc = "A directory prefix to strip from the extracted files. " +
              "Many archives contain a top-level directory that contains all of the useful files in archive.",
    ),
    "urls": attr.string_list(
        mandatory = True,
        doc = "A list of URLs to a file that will be made available to Bazel. " +
              "Each entry must be a file, http or https URL. Redirections are followed. " +
              "Authentication is not supported. " +
              "URLs are tried in order until one succeeds, so you should list local mirrors first. " +
              "If all downloads fail, the rule will fail.",
    ),
})

cuda_toolkit_tag = tag_class(attrs = {
    "name": attr.string(doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
    "components_mapping": components_mapping_compat.attr(
        doc = "A mapping from component names to component repos of a deliverable CUDA Toolkit. " +
              "Only the repo part of the label is usefull",
    ),
    "version": attr.string(),
    "nvcc_version": attr.string(),
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

def _impl(module_ctx):
    # Toolchain configuration is only allowed in the root module, or in rules_cuda.
    root, rules_cuda = _find_modules(module_ctx)
    components = None
    toolkits = None
    if root.tags.toolkit:
        components = root.tags.component
        toolkits = root.tags.toolkit
    else:
        components = rules_cuda.tags.component
        toolkits = rules_cuda.tags.toolkit

    for component in components:
        cuda_component(**_module_tag_to_dict(component))

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
    implementation = _impl,
    tag_classes = {
        "component": cuda_component_tag,
        "toolkit": cuda_toolkit_tag,
    },
)
