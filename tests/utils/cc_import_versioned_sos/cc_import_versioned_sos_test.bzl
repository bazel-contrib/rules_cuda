load("@bazel_skylib//lib:unittest.bzl", "analysistest")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//cuda/private:os_helpers.bzl", "cc_import_versioned_sos")

def _library_basenames_test_impl(ctx):
    env = analysistest.begin(ctx)

    expected_libraries = {filename: False for filename in ctx.attr.expected_libraries}

    linking_context = analysistest.target_under_test(env)[CcInfo].linking_context
    for linker_input in linking_context.linker_inputs.to_list():
        for library in linker_input.libraries:
            dynamic_library = library.dynamic_library
            if dynamic_library == None:
                continue

            if dynamic_library.basename not in expected_libraries:
                msg = "Unexpected shared library: got {} but expected one of {}.".format(
                    repr(dynamic_library.basename),
                    expected_libraries.keys(),
                )
                analysistest.fail(env, msg)
            expected_libraries[dynamic_library.basename] = True

    for filename, found in expected_libraries.items():
        if not found:
            analysistest.fail(env, "Shared library was not found among libraries to link: {}.".format(repr(filename)))

    return analysistest.end(env)

library_basenames_test = analysistest.make(
    _library_basenames_test_impl,
    attrs = {
        "expected_libraries": attr.string_list(mandatory = True),
    },
)

def _make_library_basenames_test(name, shared_library, expected_libraries):
    import_name = name + "_import"
    cc_import_versioned_sos(
        name = import_name,
        shared_library = shared_library,
    )
    library_basenames_test(
        name = name + "_test",
        expected_libraries = expected_libraries,
        target_under_test = import_name,
    )

def library_basenames_tests():
    """Test that shared library file names are not mangled after import."""

    # Test case when shared library is not installed.
    _make_library_basenames_test(
        name = "library_not_installed",
        shared_library = "liboptional.so",
        expected_libraries = [],
    )

    # Test case when shared library is located in the root of a package.
    _make_library_basenames_test(
        name = "library_in_package_root",
        shared_library = "libfoo.so",
        expected_libraries = [
            "libfoo.so",
            "libfoo.so.1",
            "libfoo.so.1.2.3",
        ],
    )

    # Test case when shared library is located in a subdirectory.
    _make_library_basenames_test(
        name = "library_in_subdirectory",
        shared_library = "subdir/libbar.so",
        expected_libraries = [
            "libbar.so",
            "libbar.so.1",
            "libbar.so.1.2.3",
        ],
    )
