#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

# Set by GH actions, see
# https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
TAG=$1
PREFIX="rules_cuda-${TAG:1}"
SHA=$(git archive --format=tar --prefix=${PREFIX}/ ${TAG} | gzip | shasum -a 256 | awk '{print $1}')

cat << EOF
## \`WORKSPACE\` code
\`\`\`starlark
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_cuda",
    sha256 = "${SHA}",
    strip_prefix = "${PREFIX}",
    url = "https://github.com/bazel-contrib/rules_cuda/archive/refs/tags/${TAG}.tar.gz",
)
EOF

awk 'f;/--SNIP--/{f=1}' examples/WORKSPACE.bazel
echo "\`\`\`" 