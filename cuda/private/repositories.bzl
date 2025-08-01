"""Generate `@cuda//`"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//cuda/private:redist_json_helper.bzl", "redist_json_helper")
load("//cuda/private:template_helper.bzl", "template_helper")
load("//cuda/private:templates/registry.bzl", "REGISTRY")
load("//cuda/private:toolchain.bzl", "register_detected_cuda_toolchains")

def _is_linux(ctx):
    return ctx.os.name.startswith("linux")

def _is_windows(ctx):
    return ctx.os.name.lower().startswith("windows")

def _get_nvcc_version(repository_ctx, nvcc_root):
    result = repository_ctx.execute([nvcc_root + "/bin/nvcc", "--version"])
    if result.return_code != 0:
        return [-1, -1]
    for line in [line for line in result.stdout.split("\n") if ", release " in line]:
        segments = line.split(", release ")
        if len(segments) != 2:
            continue
        version = [int(v) for v in segments[-1].split(", ")[0].split(".")]
        if len(version) >= 2:
            return version[:2]
    return [-1, -1]

def _detect_local_cuda_toolkit(repository_ctx):
    cuda_path = repository_ctx.attr.toolkit_path
    if cuda_path == "":
        cuda_path = repository_ctx.os.environ.get("CUDA_PATH", None)
    if cuda_path == None:
        ptxas_path = repository_ctx.which("ptxas")
        if ptxas_path:
            # ${CUDA_PATH}/bin/ptxas

            # Some distributions instead put CUDA binaries in a separate path
            # Manually check and redirect there when necessary
            alternative = repository_ctx.path("/usr/lib/nvidia-cuda-toolkit/bin/nvcc")
            if str(ptxas_path) == "/usr/bin/ptxas" and alternative.exists:
                ptxas_path = alternative
            cuda_path = str(ptxas_path.dirname.dirname)
    if cuda_path == None and _is_linux(repository_ctx):
        cuda_path = "/usr/local/cuda"

    if cuda_path != None and not repository_ctx.path(cuda_path).exists:
        cuda_path = None

    bin_ext = ".exe" if _is_windows(repository_ctx) else ""
    nvcc = "@rules_cuda//cuda/dummy:nvcc"
    nvlink = "@rules_cuda//cuda/dummy:nvlink"
    link_stub = "@rules_cuda//cuda/dummy:link.stub"
    bin2c = "@rules_cuda//cuda/dummy:bin2c"
    fatbinary = "@rules_cuda//cuda/dummy:fatbinary"
    if cuda_path != None:
        if repository_ctx.path(cuda_path + "/bin/nvcc" + bin_ext).exists:
            nvcc = str(Label("@cuda//:cuda/bin/nvcc{}".format(bin_ext)))
        if repository_ctx.path(cuda_path + "/bin/nvlink" + bin_ext).exists:
            nvlink = str(Label("@cuda//:cuda/bin/nvlink{}".format(bin_ext)))
        if repository_ctx.path(cuda_path + "/bin/crt/link.stub").exists:
            link_stub = str(Label("@cuda//:cuda/bin/crt/link.stub"))
        if repository_ctx.path(cuda_path + "/bin/bin2c" + bin_ext).exists:
            bin2c = str(Label("@cuda//:cuda/bin/bin2c{}".format(bin_ext)))
        if repository_ctx.path(cuda_path + "/bin/fatbinary" + bin_ext).exists:
            fatbinary = str(Label("@cuda//:cuda/bin/fatbinary{}".format(bin_ext)))

    nvcc_version_major = -1
    nvcc_version_minor = -1

    if cuda_path != None:
        nvcc_version_major, nvcc_version_minor = _get_nvcc_version(repository_ctx, cuda_path)

    return struct(
        path = cuda_path,
        # this should have been extracted from cuda.h, reuse nvcc for now
        version_major = nvcc_version_major,
        version_minor = nvcc_version_minor,
        # this is extracted from `nvcc --version`
        nvcc_version_major = nvcc_version_major,
        nvcc_version_minor = nvcc_version_minor,
        nvcc_label = nvcc,
        nvlink_label = nvlink,
        link_stub_label = link_stub,
        bin2c_label = bin2c,
        fatbinary_label = fatbinary,
    )

def _detect_deliverable_cuda_toolkit(repository_ctx):
    # NOTE: component nvcc contains some headers that will be used.
    required_components = ["cccl", "cudart", "nvcc"]
    for rc in required_components:
        if rc not in repository_ctx.attr.components_mapping:
            fail('component "{}" is required.'.format(rc))

    nvcc_repo = repository_ctx.attr.components_mapping["nvcc"]

    bin_ext = ".exe" if _is_windows(repository_ctx) else ""
    nvcc = "{}//:nvcc/bin/nvcc{}".format(nvcc_repo, bin_ext)
    nvlink = "{}//:nvcc/bin/nvlink{}".format(nvcc_repo, bin_ext)
    link_stub = "{}//:nvcc/bin/crt/link.stub".format(nvcc_repo)
    bin2c = "{}//:nvcc/bin/bin2c{}".format(nvcc_repo, bin_ext)
    fatbinary = "{}//:nvcc/bin/fatbinary{}".format(nvcc_repo, bin_ext)

    cuda_version_str = repository_ctx.attr.version
    if cuda_version_str == None or cuda_version_str == "":
        fail("attr version is required.")

    nvcc_version_str = repository_ctx.attr.nvcc_version
    if nvcc_version_str == None or nvcc_version_str == "":
        nvcc_version_str = cuda_version_str

    cuda_version_major, cuda_version_minor = cuda_version_str.split(".")[:2]
    nvcc_version_major, nvcc_version_minor = nvcc_version_str.split(".")[:2]

    return struct(
        path = None,  # scattered components
        version_major = cuda_version_major,
        version_minor = cuda_version_minor,
        nvcc_version_major = nvcc_version_major,
        nvcc_version_minor = nvcc_version_minor,
        nvcc_label = nvcc,
        nvlink_label = nvlink,
        link_stub_label = link_stub,
        bin2c_label = bin2c,
        fatbinary_label = fatbinary,
    )

def detect_cuda_toolkit(repository_ctx):
    """Detect CUDA Toolkit.

    The path to CUDA Toolkit is determined as:
      - use nvcc component from deliverable
      - the value of `toolkit_path` passed to `cuda_toolkit` repo rule as an attribute
      - taken from `CUDA_PATH` environment variable or
      - determined through 'which ptxas' or
      - defaults to '/usr/local/cuda'

    Args:
        repository_ctx: repository_ctx

    Returns:
        A struct contains the information of CUDA Toolkit.
    """
    if repository_ctx.attr.components_mapping != {}:
        return _detect_deliverable_cuda_toolkit(repository_ctx)
    else:
        return _detect_local_cuda_toolkit(repository_ctx)

def config_cuda_toolkit_and_nvcc(repository_ctx, cuda):
    """Generate `@cuda//BUILD` and `@cuda//defs.bzl` and `@cuda//toolchain/BUILD`

    Args:
        repository_ctx: repository_ctx
        cuda: The struct returned from detect_cuda_toolkit
    """

    # True: locally installed cuda toolkit (@cuda with full install of local CTK)
    # False: hermatic cuda toolkit (@cuda with alias of components)
    # None: cuda toolkit is not presented
    is_local_ctk = None

    if len(repository_ctx.attr.components_mapping) != 0:
        is_local_ctk = False

    if is_local_ctk == None and cuda.path != None:
        # When using a special cuda toolkit path install, need to manually fix up the lib64 links
        if cuda.path == "/usr/lib/nvidia-cuda-toolkit":
            repository_ctx.symlink(cuda.path + "/bin", "cuda/bin")
            repository_ctx.symlink("/usr/lib/x86_64-linux-gnu", "cuda/lib64")
        else:
            repository_ctx.symlink(cuda.path, "cuda")
        is_local_ctk = True

    # Generate @cuda//BUILD
    if is_local_ctk == None:
        repository_ctx.symlink(Label("//cuda/private:templates/BUILD.cuda_disabled"), "BUILD")
    elif is_local_ctk:
        libpath = "lib64" if _is_linux(repository_ctx) else "lib"
        template_helper.generate_build(repository_ctx, libpath)
    else:
        template_helper.generate_build(
            repository_ctx,
            libpath = "lib",
            components = repository_ctx.attr.components_mapping,
            is_cuda_repo = True,
            is_deliverable = True,
        )

    # Generate @cuda//defs.bzl
    template_helper.generate_defs_bzl(repository_ctx, is_local_ctk == True)

    # Generate @cuda//toolchain/BUILD
    template_helper.generate_toolchain_build(repository_ctx, cuda)

def detect_clang(repository_ctx):
    """Detect local clang installation.

    The path to clang is determined by:

      - taken from configured cc_toolchain if `CUDA_COMPILER_USE_CC_TOOLCHAIN` = "true"
      - taken from `CUDA_CLANG_LABEL` environment variable
      - taken from `CUDA_CLANG_PATH` environment variable
      - taken from `BAZEL_LLVM` environment variable as `$BAZEL_LLVM/bin/clang` or
      - determined through `which clang` or
      - treated as being not detected and not configured

    Args:
        repository_ctx: repository_ctx

    Returns:
        clang_path_or_label (str | None): Optionally return a string of path or label to clang executable if detected.
    """
    bin_ext = ".exe" if _is_windows(repository_ctx) else ""

    clang_path = repository_ctx.os.environ.get("CUDA_CLANG_PATH", None)
    clang_label = repository_ctx.os.environ.get("CUDA_CLANG_LABEL", None)
    clang_path_or_label = None

    if clang_label != None:
        # Check CUDA_CLANG_LABEL first
        clang_path_or_label = clang_label

    elif clang_path != None and repository_ctx.path(clang_path).exists:
        # Check CUDA_CLANG_PATH next
        clang_path_or_label = clang_path

    else:
        # Check BAZEL_LLVM
        bazel_llvm = repository_ctx.os.environ.get("BAZEL_LLVM", None)
        if bazel_llvm != None and repository_ctx.path(bazel_llvm + "/bin/clang" + bin_ext).exists:
            clang_path_or_label = bazel_llvm + "/bin/clang" + bin_ext
        elif repository_ctx.which("clang") != None:
            # Finally try 'which clang'
            clang_path_or_label = str(repository_ctx.which("clang"))

    return clang_path_or_label

def config_clang(repository_ctx, cuda, clang_path_or_label):
    """Generate `@cuda//toolchain/clang/BUILD`

    Args:
        repository_ctx: repository_ctx
        cuda: The struct returned from `detect_cuda_toolkit`
        clang_path_or_label: Path or label to clang executable returned from `detect_clang`
    """
    is_local_ctk = True

    if len(repository_ctx.attr.components_mapping) != 0:
        is_local_ctk = False

    # Generate @cuda//toolchain/clang/BUILD
    template_helper.generate_toolchain_clang_build(repository_ctx, cuda, clang_path_or_label)

def config_disabled(repository_ctx):
    repository_ctx.symlink(Label("//cuda/private:templates/BUILD.toolchain_disabled"), "toolchain/disabled/BUILD")

def _cuda_toolkit_impl(repository_ctx):
    cuda = detect_cuda_toolkit(repository_ctx)
    config_cuda_toolkit_and_nvcc(repository_ctx, cuda)

    clang_path_or_label = detect_clang(repository_ctx)
    config_clang(repository_ctx, cuda, clang_path_or_label)

    config_disabled(repository_ctx)

cuda_toolkit = repository_rule(
    implementation = _cuda_toolkit_impl,
    attrs = {
        "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
        "components_mapping": attr.string_dict(
            doc = "A mapping from component names to component repos of a deliverable CUDA Toolkit. " +
                  "Only the repo part of the label is useful",
        ),
        "version": attr.string(doc = "cuda toolkit version. Required for deliverable toolkit only."),
        "nvcc_version": attr.string(
            doc = "nvcc version. Required for deliverable toolkit only. Fallback to version if omitted.",
        ),
    },
    configure = True,
    local = True,
    environ = ["CUDA_PATH", "PATH", "CUDA_CLANG_PATH", "CUDA_CLANG_LABEL", "BAZEL_LLVM", "CUDA_COMPILER_USE_CC_TOOLCHAIN"],
    # remotable = True,
)

def _cuda_component_impl(repository_ctx):
    component_name = None
    if repository_ctx.attr.component_name:
        component_name = repository_ctx.attr.component_name
        if component_name not in REGISTRY:
            fail("invalid component '{}', available: {}".format(component_name, repr(REGISTRY.keys())))
    else:
        component_name = repository_ctx.name[len("cuda_"):]
        if component_name not in REGISTRY:
            fail("invalid derived component '{}', available: {}, ".format(component_name, repr(REGISTRY.keys())) +
                 " if derivation result is unexpected, please specify `component_name` attribute manually")

    if not repository_ctx.attr.url and not repository_ctx.attr.urls:
        fail("either attribute `url` or `urls` must be filled")
    if repository_ctx.attr.url and repository_ctx.attr.urls:
        fail("attributes `url` and `urls` cannot be used at the same time")

    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url or repository_ctx.attr.urls,
        output = component_name,
        integrity = repository_ctx.attr.integrity,
        sha256 = repository_ctx.attr.sha256,
        stripPrefix = repository_ctx.attr.strip_prefix,
    )

    template_helper.generate_build(
        repository_ctx,
        libpath = "lib",
        components = {component_name: repository_ctx.name},
        is_cuda_repo = False,
        is_deliverable = True,
    )

    desc_name = repository_ctx.attr.descriptive_name or repository_ctx.attr.component_name
    repository_ctx.file(
        "{}/version.json".format(component_name),
        content = json.encode({
            component_name: {
                "name": desc_name,
                "version": repository_ctx.attr.version,
            },
        }),
    )

    # redistrib_<version>.json have fields of "license_path" which can be accessed
    # For example https://developer.download.nvidia.cn/compute/cuda/redist/redistrib_12.8.1.json
    #         "license_path": "cuda_cccl/LICENSE.txt"
    # can be accessed from https://developer.download.nvidia.cn/compute/cuda/redist/cuda_nvcc/LICENSE.txt
    license_path = repository_ctx.path("{}/LICENSE".format(component_name))
    if not license_path.exists:
        repository_ctx.file(
            license_path,
            content = (
                "Deliverable archive download from " + repr(repository_ctx.attr.url or repository_ctx.attr.urls) +
                " does not provides a LICENSE file.\n\nPlease consult deliverable archive provider for details."
            ),
        )

cuda_component = repository_rule(
    implementation = _cuda_component_impl,
    attrs = {
        "component_name": attr.string(doc = "Short name of the component defined in registry."),
        "descriptive_name": attr.string(doc = "Official name of a component or simply the component name."),
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
        "version": attr.string(doc = "A unique version number for component. Store in version.json file"),
    },
)

def default_components_mapping(components):
    """Create a default components_mapping from list of component names.

    Args:
        components: list of string, a list of component names.
    """
    return {c: "@cuda_" + c for c in components}

def _cuda_redist_json_impl(repository_ctx):
    attr = repository_ctx.attr
    url, json_object = redist_json_helper.get(repository_ctx, attr)
    redist_ver = redist_json_helper.get_redist_version(repository_ctx, attr, json_object)
    specs = redist_json_helper.collect_specs(repository_ctx, attr, json_object, url)

    template_helper.generate_redist_bzl(repository_ctx, specs, redist_ver)
    repository_ctx.symlink(Label("//cuda/private:templates/BUILD.redist_json"), "BUILD")

cuda_redist_json = repository_rule(
    implementation = _cuda_redist_json_impl,
    attrs = {
        "components": attr.string_list(mandatory = True),
        "integrity": attr.string(mandatory = False),
        "sha256": attr.string(mandatory = False),
        "urls": attr.string_list(mandatory = False),
        "version": attr.string(mandatory = False),
    },
)

def rules_cuda_dependencies():
    """Populate the dependencies for rules_cuda. This will setup other bazel rules as workspace dependencies"""
    maybe(
        name = "bazel_skylib",
        repo_rule = http_archive,
        sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        ],
    )

    maybe(
        name = "platforms",
        repo_rule = http_archive,
        sha256 = "379113459b0feaf6bfbb584a91874c065078aa673222846ac765f86661c27407",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
        ],
    )

def rules_cuda_toolchains(toolkit_path = None, components_mapping = None, version = None, nvcc_version = None, register_toolchains = False):
    """Populate the @cuda repo.

    Args:
        toolkit_path: Optionally specify the path to CUDA toolkit. If not specified, it will be detected automatically.
        components_mapping: dict mapping from component_name to its corresponding cuda_component's repo_name
        version: str for cuda toolkit version. Required for deliverable toolkit only.
        nvcc_version: str for nvcc version. Required for deliverable toolkit only. Fallback to version if omitted.
        register_toolchains: Register the toolchains if enabled.
    """

    cuda_toolkit(
        name = "cuda",
        toolkit_path = toolkit_path,
        components_mapping = components_mapping,
        version = version,
        nvcc_version = nvcc_version,
    )

    if register_toolchains:
        register_detected_cuda_toolchains()
