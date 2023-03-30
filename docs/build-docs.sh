#!/bin/bash

function prepare-env {
    pip install -r requirements.txt
}

function compose-docs {
    rm -rf docs site
    mkdir -p docs
    cp ../README.md docs/index.md
    bazel build :all_docs
    rsync -a --prune-empty-dirs --include '*/' mkdocs/stylesheets docs/
    rsync -a --prune-empty-dirs --include '*/' --include '*.md' --exclude '*' bazel-bin/ docs/
    find docs/ -name '*.md' -exec sed -i 's#<pre>#<div class="stardoc-pre"><pre>#g' {} \;
    find docs/ -name '*.md' -exec sed -i 's#</pre>#</pre></div>#g' {} \;
}

function compose-versioned-site {
    mkdir -p generated
    rsync -a --prune-empty-dirs --include '*/' site/ generated/$1/
    python versioning.py generated/ --force

    printf "\nRun following command to update version list then serve locally:\n\n"
    printf "\tpython -m http.server -d generated/\n\n"
}


CI=${CI:-0}  # 1 for CI only logic

if [ $CI == "1" ]; then
    set -ex
    prepare-env
    compose-docs
    mkdocs build
else
    if [[ $# -ne 1 ]]; then
        printf "Usage: $0 <version>\n"
        exit -1
    fi
    version=$1

    # env should be prepared manually
    compose-docs
    mkdocs build
    compose-versioned-site $version
fi
