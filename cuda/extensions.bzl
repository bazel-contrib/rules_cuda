"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "cuda_component", "cuda_redist_json", "local_cuda")

cuda_component_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True, doc = "Repo name for the deliverable cuda_component"),
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
    "name": attr.string(doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
    "components": attr.string_list(doc = "Component names of a deliverable CUDA Toolkit."),
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

def _impl(module_ctx):
    # Toolchain configuration is only allowed in the root module, or in rules_cuda.
    root, rules_cuda = _find_modules(module_ctx)
    components = None
    redist_jsons = None
    toolchains = None
    if root.tags.local_toolchain:
        components = root.tags.component
        redist_jsons = root.tags.redist_json
        toolchains = root.tags.local_toolchain
    else:
        components = rules_cuda.tags.component
        redist_jsons = rules_cuda.tags.redist_json
        toolchains = rules_cuda.tags.local_toolchain

    for component in components:
        cuda_component(
            name = component.name,
            integrity = component.integrity,
            sha256 = component.sha256,
            strip_prefix = component.strip_prefix,
            urls = component.urls,
        )

    for redist_json in redist_jsons:
        cuda_redist_json(
            name = redist_json.name,
            components = redist_json.components,
            integrity = redist_json.integrity,
            sha256 = redist_json.sha256,
            urls = redist_json.urls,
            version = redist_json.version,
        )

    registrations = {}
    for toolchain in toolchains:
        if toolchain.name in registrations.keys():
            # FIXME: how to handle components?
            if toolchain.toolkit_path == registrations[toolchain.name]:
                # No problem to register a matching toolchain twice
                continue
            fail("Multiple conflicting toolchains declared for name {} ({} and {}".format(toolchain.name, toolchain.toolkit_path, registrations[toolchain.name]))
        else:
            registrations[toolchain.name] = toolchain.toolkit_path
    for name, toolkit_path in registrations.items():
        local_cuda(name = name, toolkit_path = toolkit_path)

toolchain = module_extension(  # TODO: rename toolchain to cuda
    implementation = _impl,
    tag_classes = {
        "component": cuda_component_tag,
        "redist_json": cuda_redist_json_tag,
        "local_toolchain": cuda_toolkit_tag,  # TODO: rename local_toolchain to toolkit
    },
)
