load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:redist_json_helper.bzl", "redist_json_helper")

_URL = "https://example.com/redist/redistrib_13.4.2.json"

def _entry(name):
    return {
        "name": name,
        "version": "13.4.2",
        "linux-x86_64": {
            "relative_path": "{0}/linux-x86_64/{0}-linux-x86_64-13.4.2-archive.tar.xz".format(name),
            "sha256": "00",
        },
    }

def _collect_specs_test_impl(ctx):
    env = unittest.begin(ctx)

    # CUDA 13.3+ manifests key CCCL as "cccl", older ones as "cuda_cccl".
    for key in ["cuda_cccl", "cccl"]:
        redist = {key: _entry("CCCL"), "cuda_nvcc": _entry("CUDA NVCC")}
        for components in [["cccl"], []]:
            specs = redist_json_helper.collect_specs(ctx, struct(components = components), "linux-x86_64", redist, _URL)
            names = sorted([s["component_name"] for s in specs])
            asserts.equals(env, ["cccl", "nvcc"] if not components else ["cccl"], names)

    return unittest.end(env)

_collect_specs_test = unittest.make(_collect_specs_test_impl)

def collect_specs_test(name):
    unittest.suite(name, _collect_specs_test)
