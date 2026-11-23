"""Tests for cc_import_static_and_versioned_so."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//cuda/private:os_helpers.bzl", "cc_import_static_and_versioned_so")

def _static_and_versioned_so_test_impl(ctx):
    env = analysistest.begin(ctx)

    static_libraries = []
    dynamic_libraries = []
    linking_context = analysistest.target_under_test(env)[CcInfo].linking_context
    for linker_input in linking_context.linker_inputs.to_list():
        for library in linker_input.libraries:
            if library.static_library:
                static_libraries.append(library.static_library.basename)
            if library.pic_static_library:
                static_libraries.append(library.pic_static_library.basename)
            if library.dynamic_library:
                dynamic_libraries.append(library.dynamic_library.basename)

    asserts.equals(env, ctx.attr.expected_static_library, static_libraries[0] if static_libraries else "")
    asserts.equals(env, ctx.attr.expected_dynamic_libraries, dynamic_libraries)

    return analysistest.end(env)

static_and_versioned_so_test = analysistest.make(
    _static_and_versioned_so_test_impl,
    attrs = {
        "expected_dynamic_libraries": attr.string_list(),
        "expected_static_library": attr.string(mandatory = True),
    },
)

def _make_test(name, static_library, shared_library, expected_static_library, expected_dynamic_libraries):
    import_name = name + "_import"
    cc_import_static_and_versioned_so(
        name = import_name,
        static_library = static_library,
        shared_library = shared_library,
    )
    static_and_versioned_so_test(
        name = name + "_test",
        expected_dynamic_libraries = expected_dynamic_libraries,
        expected_static_library = expected_static_library,
        target_under_test = import_name,
    )

def static_and_versioned_so_tests():
    """Test that the static library and the SONAME-named shared library are imported."""

    # The shortest versioned file is selected as the shared library.
    _make_test(
        name = "selects_soname",
        static_library = "libfoo.a",
        shared_library = "libfoo.so",
        expected_static_library = "libfoo.a",
        expected_dynamic_libraries = ["libfoo.so.1"],
    )

    # Only the static library is imported when no versioned shared library exists.
    _make_test(
        name = "static_only",
        static_library = "libstatic_only.a",
        shared_library = "libstatic_only.so",
        expected_static_library = "libstatic_only.a",
        expected_dynamic_libraries = [],
    )
