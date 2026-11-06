load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:redist_json_helper.bzl", "redist_json_helper")
load("//cuda/private:templates/registry.bzl", "FULL_COMPONENT_NAME", "get_full_component_name")
load(":manifest_component_names.bzl", "MANIFEST_COMPONENT_NAMES")

_URL = "https://example.com/redist/redistrib.json"
_PLATFORM = "linux-x86_64"
_COMPONENT_VERSION = "3.4.5"

def _entry(name, platform = _PLATFORM):
    return {
        "name": name,
        "version": _COMPONENT_VERSION,
        platform: {
            "relative_path": "{0}/{1}/{0}-{1}-{2}-archive.tar.xz".format(name, platform, _COMPONENT_VERSION),
            "sha256": "00",
        },
    }

def _repository_ctx(environ = {}):
    return struct(os = struct(environ = environ))

def _expected_spec(component, name):
    archive = "{}-{}-{}-archive".format(name, _PLATFORM, _COMPONENT_VERSION)
    return {
        "component_name": component,
        "descriptive_name": name,
        "urls": ["https://example.com/redist/{}/{}/{}.tar.xz".format(name, _PLATFORM, archive)],
        "sha256": "00",
        "strip_prefix": archive,
        "version": _COMPONENT_VERSION,
    }

def _component_name_test_impl(ctx):
    env = unittest.begin(ctx)

    for versions, names in MANIFEST_COMPONENT_NAMES:
        for version in versions:
            for component, name in names.items():
                asserts.equals(env, name, get_full_component_name(component, version), "{} at {}".format(component, version))

    # Bounds are exclusive and compare numeric versions, including future releases.
    for version in ["11.4.4", "12.9.2", "13.0.0", "13.2.999"]:
        asserts.equals(env, "cuda_cccl", get_full_component_name("cccl", version))
    for version in ["13.3", "13.3.0", "13.3.1", "13.10.0", "14.0.0", "latest"]:
        asserts.equals(env, "cccl", get_full_component_name("cccl", version))

    # Components without renames retain their names beyond the audited releases.
    for component, name in MANIFEST_COMPONENT_NAMES[-1][1].items():
        asserts.equals(env, name, get_full_component_name(component, "latest"))
        asserts.equals(env, name, get_full_component_name(component, "100.0.0"))

    return unittest.end(env)

def _collect_specs_test_impl(ctx):
    env = unittest.begin(ctx)

    # Test every audited release without downloading toolkits or requiring a GPU.
    for versions, names in MANIFEST_COMPONENT_NAMES:
        for version in versions:
            redist = {name: _entry(name) for name in names.values()}
            redist["release_label"] = version
            redist["unrelated_component"] = _entry("unrelated_component")
            expected = {c: _expected_spec(c, name) for c, name in names.items()}
            for components in [[], names.keys(), FULL_COMPONENT_NAME.keys()]:
                specs = redist_json_helper.collect_specs(_repository_ctx(), struct(components = components), _PLATFORM, redist, _URL)
                asserts.equals(env, len(expected), len(specs), version)
                asserts.equals(env, expected, {s["component_name"]: s for s in specs}, version)
            for component in names:
                specs = redist_json_helper.collect_specs(_repository_ctx(), struct(components = [component]), _PLATFORM, redist, _URL)
                asserts.equals(env, [expected[component]], specs, version)

    return unittest.end(env)

def _version_source_test_impl(ctx):
    env = unittest.begin(ctx)

    # Both aliases may exist: select by toolkit version, never by key presence or
    # the component's own version. Include explicit versions for old manifests
    # without release_label, and the version override used by CI.
    for release_label, attr_version, override, expected_name in [
        ("13.2.2", "", "", "cuda_cccl"),
        ("13.3.0", "", "", "cccl"),
        (None, "11.8.0", "", "cuda_cccl"),
        (None, "13.3.0", "", "cccl"),
        ("13.2.2", "13.3.0", "", "cccl"),
        ("13.2.2", "13.2.2", "13.3.0", "cccl"),
        ("13.3.0", "13.3.0", "13.2.2", "cuda_cccl"),
    ]:
        redist = {name: _entry(name) for name in ["cuda_cccl", "cccl"]}
        if release_label != None:
            redist["release_label"] = release_label
        repository_ctx = _repository_ctx({"CUDA_REDIST_VERSION_OVERRIDE": override})
        for components in [["cccl"], []]:
            attr = struct(components = components, version = attr_version)
            specs = redist_json_helper.collect_specs(repository_ctx, attr, _PLATFORM, redist, _URL)
            asserts.equals(env, [_expected_spec("cccl", expected_name)], specs)

    return unittest.end(env)

def _platform_filter_test_impl(ctx):
    env = unittest.begin(ctx)

    redist = {
        "release_label": "13.3.0",
        "cccl": _entry("cccl"),
        "cuda_nvcc": _entry("cuda_nvcc", "windows-x86_64"),
        "cuda_cccl": _entry("cuda_cccl", "windows-x86_64"),
    }
    for components in [[], ["cccl", "nvcc", "nvjitlink"]]:
        attr = struct(components = components)
        specs = redist_json_helper.collect_specs(_repository_ctx(), attr, _PLATFORM, redist, _URL)
        asserts.equals(env, [_expected_spec("cccl", "cccl")], specs)
        specs = redist_json_helper.collect_specs(_repository_ctx(), attr, "windows-x86_64", redist, _URL)
        asserts.equals(env, ["nvcc"], [s["component_name"] for s in specs])
        specs = redist_json_helper.collect_specs(_repository_ctx(), attr, "linux-sbsa", redist, _URL)
        asserts.equals(env, [], specs)

    return unittest.end(env)

_component_name_test = unittest.make(_component_name_test_impl)
_collect_specs_test = unittest.make(_collect_specs_test_impl)
_version_source_test = unittest.make(_version_source_test_impl)
_platform_filter_test = unittest.make(_platform_filter_test_impl)

def collect_specs_test(name):
    unittest.suite(name, _component_name_test, _collect_specs_test, _version_source_test, _platform_filter_test)
