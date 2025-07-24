load("//cuda/private:templates/registry.bzl", "FULL_COMPONENT_NAME")

def _is_linux(ctx):
    return ctx.os.name.startswith("linux")

def _is_windows(ctx):
    return ctx.os.name.lower().startswith("windows")

def _get(ctx, attr):
    """Download the redistrib_<version>.json file.

    Args:
        ctx: repository_ctx | module_ctx
        attr: cuda_component repo attr or cuda_redist_json_tag

    Returns:
        (url, json_object)
    """

    the_url = None  # the url that successfully fetch redist json, we then use it to fetch deliverables
    urls = [u for u in attr.urls]

    redist_ver = attr.version
    if redist_ver:
        urls.append("https://developer.download.nvidia.com/compute/cuda/redist/redistrib_{}.json".format(redist_ver))

    if len(urls) == 0:
        fail("`urls` or `version` must be specified.")

    for url in urls:
        ret = ctx.download(
            output = "redist.json",
            integrity = attr.integrity,
            sha256 = attr.sha256,
            url = url,
        )
        if ret.success:
            the_url = url
            break

    if the_url == None:
        fail("Failed to retrieve the redist json file.")

    return the_url, json.decode(ctx.read("redist.json"))

def _get_redist_version(ctx, attr, redist):
    """Get version string.

    Args:
        ctx: repository_ctx | module_ctx
        attr: cuda_component repo attr or cuda_redist_json_tag
        redist: json object, read from the redistrib_<version>.json file.
    """

    redist_ver = attr.version
    if not redist_ver:
        redist_ver = redist["release_label"]

    return redist_ver

def _collect_specs(ctx, attr, redist, the_url):
    """Convert redistrib_<version>.json content to the specs for instantiating cuda_component repos.

    List of specs, aka, list of dicts with cuda_component attrs.

    Args:
        ctx: repository_ctx | module_ctx
        attr: cuda_component repo attr or cuda_redist_json_tag
        redist: json object, read from the redistrib_<version>.json file.
        the_url: string, the very unique url from which we get the redistrib_<version>.json file.
    """

    specs = []
    os = None
    if _is_linux(ctx):
        os = "linux"
    elif _is_windows(ctx):
        os = "windows"

    arch = "x86_64"  # TODO: support cross compiling
    platform = "{os}-{arch}".format(os = os, arch = arch)
    components = attr.components if attr.components else [k for k, v in FULL_COMPONENT_NAME.items() if v in redist]

    for c in components:
        c_full = FULL_COMPONENT_NAME[c]

        payload = redist[c_full][platform]
        payload_relative_path = payload["relative_path"]
        payload_url = the_url.rsplit("/", 1)[0] + "/" + payload_relative_path
        archive_name = payload_relative_path.rsplit("/", 1)[1].split("-archive.")[0] + "-archive"
        desc_name = redist[c_full].get("name", c_full)

        specs.append({
            "component_name": c,
            "descriptive_name": desc_name,
            "urls": [payload_url],
            "sha256": payload["sha256"],
            "strip_prefix": archive_name,
            "version": redist[c_full]["version"],
        })

    return specs

def _get_repo_name(ctx, spec):
    """Get cannonical repo name when using redistrib_<version>.json file.

    Args:
        ctx: repository_ctx | module_ctx
        spec: cuda_component attrs
    """

    repo_name = "cuda_" + spec["component_name"]
    version = spec.get("version", None)
    if version != None:
        repo_name = repo_name + "_v" + version

    return repo_name

redist_json_helper = struct(
    get = _get,
    get_redist_version = _get_redist_version,
    collect_specs = _collect_specs,
    get_repo_name = _get_repo_name,
)
