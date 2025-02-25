"""Entry point for extensions used by bzlmod."""

load("//cuda/private:compat.bzl", "components_mapping_compat")
load("//cuda/private:repositories.bzl", "cuda_component", "cuda_redist_json", "local_cuda")

cuda_component_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True, doc = "Repo name for the deliverable cuda_component"),
    "component_name": attr.string(doc = "Short name of the component defined in registry."),
    "integrity": attr.string(
        doc = "Expected checksum in Subresource Integrity format of the file downloaded. " +
              "This must match the checksum of the file downloaded.",
    ),
    "sha256": attr.string(
        doc = "The expected SHA-256 of the file downloaded. This must match the SHA-256 of the file downloaded.",
    ),
    "strip_prefix": attr.string(
        doc = "A directory prefix to strip from the extracted files. " +
              "Many archives contain a top-level directory that contains all of the useful files in archive.",
    ),
    "url": attr.string(
        doc = "A URL to a file that will be made available to Bazel. " +
              "This must be a file, http or https URL." +
              "Redirections are followed. Authentication is not supported. " +
              "More flexibility can be achieved by the urls parameter that allows " +
              "to specify alternative URLs to fetch from.",
    ),
    "urls": attr.string_list(
        doc = "A list of URLs to a file that will be made available to Bazel. " +
              "Each entry must be a file, http or https URL. " +
              "Redirections are followed. Authentication is not supported. " +
              "URLs are tried in order until one succeeds, so you should list local mirrors first. " +
              "If all downloads fail, the rule will fail.",
    ),
})

cuda_redist_json_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True, doc = "Repo name for the cuda_redist_json"),
    "components": attr.string_list(mandatory = True, doc = "components to be used"),
    "integrity": attr.string(
        doc = "Expected checksum in Subresource Integrity format of the file downloaded. " +
              "This must match the checksum of the file downloaded.",
    ),
    "sha256": attr.string(
        doc = "The expected SHA-256 of the file downloaded. " +
              "This must match the SHA-256 of the file downloaded.",
    ),
    "urls": attr.string_list(
        doc = "A list of URLs to a file that will be made available to Bazel. " +
              "Each entry must be a file, http or https URL. Redirections are followed. " +
              "Authentication is not supported. " +
              "URLs are tried in order until one succeeds, so you should list local mirrors first. " +
              "If all downloads fail, the rule will fail.",
    ),
    "version": attr.string(
        doc = "Generate a URL by using the specified version." +
              "This URL will be tried after all URLs specified in the `urls` attribute.",
    ),
})

cuda_toolkit_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True, doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(
        doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path.",
    ),
    "components_mapping": components_mapping_compat.attr(
        doc = "A mapping from component names to component repos of a deliverable CUDA Toolkit. " +
              "Only the repo part of the label is usefull",
    ),
    "version": attr.string(doc = "cuda toolkit version. Required for deliverable toolkit only."),
    "nvcc_version": attr.string(
        doc = "nvcc version. Required for deliverable toolkit only. Fallback to version if omitted.",
    ),
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
    redist_jsons = None
    toolkits = None
    if root.tags.toolkit:
        components = root.tags.component
        redist_jsons = root.tags.redist_json
        toolkits = root.tags.toolkit
    else:
        components = rules_cuda.tags.component
        redist_jsons = rules_cuda.tags.redist_json
        toolkits = rules_cuda.tags.toolkit

    for component in components:
        cuda_component(**_module_tag_to_dict(component))

    for redist_json in redist_jsons:
        cuda_redist_json(**_module_tag_to_dict(redist_json))

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
        "redist_json": cuda_redist_json_tag,
        "toolkit": cuda_toolkit_tag,
    },
)
