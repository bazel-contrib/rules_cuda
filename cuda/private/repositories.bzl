load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _local_cuda_impl(repository_ctx):
    # Path to CUDA Toolkit is
    # - taken from CUDA_PATH environment variable or
    # - determined through 'which ptxas' or
    # - defaults to '/usr/local/cuda'
    cuda_path = "/usr/local/cuda"
    ptxas_path = repository_ctx.which("ptxas")
    if ptxas_path:
        cuda_path = ptxas_path.dirname.dirname
    cuda_path = repository_ctx.os.environ.get("CUDA_PATH", cuda_path)

    defs_template = "def if_local_cuda(true, false = []):\n    return %s"
    if repository_ctx.path(cuda_path).exists:
        repository_ctx.symlink(cuda_path, "cuda")
        repository_ctx.symlink(Label("//cuda:runtime/BUILD.local_cuda"), "BUILD")
        repository_ctx.file("defs.bzl", defs_template % "true")
    else:
        repository_ctx.file("BUILD")  # Empty file
        repository_ctx.file("defs.bzl", defs_template % "false")

_local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    environ = ["CUDA_PATH", "PATH"],
    # remotable = True,
)

def rules_cuda_deps():
    maybe(
        name = "bazel_skylib",
        repo_rule = http_archive,
        sha256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
        ],
    )

    maybe(
        name = "platforms",
        repo_rule = http_archive,
        sha256 = "079945598e4b6cc075846f7fd6a9d0857c33a7afc0de868c2ccb96405225135d",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.4/platforms-0.0.4.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.4/platforms-0.0.4.tar.gz",
        ],
    )
    _local_cuda(name = "local_cuda")
