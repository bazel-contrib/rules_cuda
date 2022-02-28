load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:toolchain_config_lib.bzl", "access", "create_var_from_value", "eval_flag_group", "exist", "expand_flag", "flag_group", "parse_flag", "variable_with_value")

def _parse_flag_test_impl(ctx):
    env = unittest.begin(ctx)
    f = parse_flag("")
    asserts.equals(env, [], f.chunks)
    asserts.equals(env, {}, f.expandables)

    # f = parse_flag("%")
    # asserts.expect_failure(env, "expected '{'")

    # f = parse_flag("%{")
    # asserts.expect_failure(env, "expected variable name")

    # f = parse_flag("%{}")
    # asserts.expect_failure(env, "expected variable name")

    # f = parse_flag("%{v1.v2 ")
    # asserts.expect_failure(env, "expected '}'")

    f = parse_flag("%%")
    asserts.equals(env, ["%"], f.chunks)
    asserts.equals(env, {}, f.expandables)

    f = parse_flag("%{v}")
    asserts.equals(env, ["v"], f.chunks)
    asserts.equals(env, {"v": [0]}, f.expandables)

    f = parse_flag(" %{v}")
    asserts.equals(env, [" ", "v"], f.chunks)
    asserts.equals(env, {"v": [1]}, f.expandables)

    f = parse_flag("%%{var}")
    asserts.equals(env, ["%{var}"], f.chunks)
    asserts.equals(env, {}, f.expandables)

    f = parse_flag(" %{v1} %{v2.v3} ")
    asserts.equals(env, [" ", "v1", " ", "v2.v3", " "], f.chunks)
    asserts.equals(env, {"v1": [1], "v2.v3": [3]}, f.expandables)

    f = parse_flag("%{v1} %{v2.v3}  --flag %{v1}")
    asserts.equals(env, ["v1", " ", "v2.v3", "  --flag ", "v1"], f.chunks)
    asserts.equals(env, {"v1": [0, 4], "v2.v3": [2]}, f.expandables)

    return unittest.end(env)

parse_flag_test = unittest.make(_parse_flag_test_impl)

def _parse_flag_cache_test_impl(ctx):
    env = unittest.begin(ctx)
    cache = {}
    flag_str = "%{v}"
    f1 = parse_flag(flag_str, cache = cache)
    asserts.equals(env, ["v"], f1.chunks)
    asserts.equals(env, {"v": [0]}, f1.expandables)
    asserts.equals(env, ["v"], cache[flag_str].chunks)
    asserts.equals(env, {"v": [0]}, cache[flag_str].expandables)
    f1.chunks.append("modifed")
    f1.expandables["v"].append("modifed")
    asserts.equals(env, ["v"], cache[flag_str].chunks)
    asserts.equals(env, {"v": [0]}, cache[flag_str].expandables)
    f2 = parse_flag(flag_str, cache = cache)
    asserts.equals(env, ["v"], f2.chunks)
    asserts.equals(env, {"v": [0]}, f2.expandables)
    asserts.equals(env, ["v"], cache[flag_str].chunks)
    asserts.equals(env, {"v": [0]}, cache[flag_str].expandables)

    return unittest.end(env)

parse_flag_cache_test = unittest.make(_parse_flag_cache_test_impl)

def create_var(var, path = None, path_list = None):
    if path_list == None:
        path_list = path.split(".")
    v = access(var, path_list = path_list)
    return create_var_from_value(var, v, path, path_list)

def _var_test_impl(ctx):
    env = unittest.begin(ctx)
    inner_var = struct(inner = "value")
    outer_var = struct(struct = inner_var)

    asserts.false(env, exist(outer_var, "not_exist"))
    asserts.true(env, exist(outer_var, "struct"))
    asserts.true(env, exist(outer_var, "struct.inner"))

    asserts.equals(env, inner_var, access(outer_var, "struct"))
    asserts.equals(env, "value", access(outer_var, "struct.inner"))

    asserts.equals(env, inner_var, create_var(inner_var, "inner"))
    asserts.equals(env, outer_var, create_var(outer_var, "struct"))
    asserts.equals(env, outer_var, create_var(outer_var, "struct.inner"))
    return unittest.end(env)

var_test = unittest.make(_var_test_impl)

def _expand_flag_test_impl(ctx):
    env = unittest.begin(ctx)
    f = parse_flag("--flag=%{v}")
    expand_flag(f, struct(v = "the-value"), "v")
    asserts.equals(env, ["--flag=", "the-value"], f.chunks)
    asserts.equals(env, {}, f.expandables)

    f = parse_flag("--flag=%{v}")
    expand_flag(f, struct(v = "the-value"), "not_exist")
    asserts.equals(env, ["--flag=", "v"], f.chunks)
    asserts.equals(env, {"v": [1]}, f.expandables)

    f = parse_flag("--flag=%{v1}x%{v2}")
    var = struct(v1 = "v1-value", v2 = "v2-value")
    expand_flag(f, var, "v2")
    asserts.equals(env, ["--flag=", "v1", "x", "v2-value"], f.chunks)
    asserts.equals(env, {"v1": [1]}, f.expandables)
    expand_flag(f, var, "v1")
    asserts.equals(env, ["--flag=", "v1-value", "x", "v2-value"], f.chunks)
    asserts.equals(env, {}, f.expandables)

    return unittest.end(env)

expand_flag_test = unittest.make(_expand_flag_test_impl)

def _eval_flag_group_test_impl(ctx):
    env = unittest.begin(ctx)

    # https://github.com/bazelbuild/bazel/blob/ac48e65f70/tools/cpp/cc_toolchain_config_lib.bzl#L204-L209
    fg = flag_group(
        iterate_over = "include_path",
        flags = ["-I", "%{include_path}"],
    )
    st = struct(include_path = ["/to/path1", "/to/path2", "/to/path3"])
    asserts.equals(env, ["-I", "/to/path1", "-I", "/to/path2", "-I", "/to/path3"], eval_flag_group(fg, st, 128))

    # https://github.com/bazelbuild/bazel/blob/ac48e65f70/tools/cpp/cc_toolchain_config_lib.bzl#L212-L220
    fg = flag_group(
        iterate_over = "libraries_to_link",
        flag_groups = [
            flag_group(
                iterate_over = "libraries_to_link.libraries",
                flags = ["-L%{libraries_to_link.libraries.directory}"],
            ),
        ],
    )
    st = struct(libraries_to_link = [
        struct(libraries = [struct(directory = "lib1/dir1"), struct(directory = "lib1/dir2")]),
        struct(libraries = [struct(directory = "lib2/dir1"), struct(directory = "lib2/dir2")]),
    ])
    asserts.equals(env, ["-Llib1/dir1", "-Llib1/dir2", "-Llib2/dir1", "-Llib2/dir2"], eval_flag_group(fg, st, 128))

    # https://github.com/bazelbuild/bazel/blob/ac48e65f70/tools/cpp/cc_toolchain_config_lib.bzl#L226-L244
    fg = flag_group(
        iterate_over = "object_files",
        flag_groups = [
            flag_group(flags = ["--start-lib"]),
            flag_group(iterate_over = "object_files", flags = ["%{object_files}"]),
            flag_group(flags = ["--end-lib"]),
        ],
    )
    st = struct(object_files = [["a1.o", "a2.o"], ["b1.o", "b2.o"]])
    asserts.equals(env, ["--start-lib", "a1.o", "a2.o", "--end-lib", "--start-lib", "b1.o", "b2.o", "--end-lib"], eval_flag_group(fg, st, 128))

    # Following test cases are coming from
    # https://github.com/bazelbuild/bazel/blob/ae79934217/src/test/java/com/google/devtools/build/lib/rules/cpp/CcToolchainFeaturesTest.java

    fg = flag_group(flag_groups = [flag_group(flags = ["-A%{struct.foo}"]), flag_group(flags = ["-B%{struct.bar}"])])
    st = struct(struct = struct(foo = "fooValue", bar = "barValue"))
    asserts.equals(env, ["-AfooValue", "-BbarValue"], eval_flag_group(fg, st, 128), "testSimpleStructureVariableExpansion")

    fg = flag_group(flags = ["-A%{struct.foo.bar}"])
    st = struct(struct = struct(foo = struct(bar = "fooBarValue")))
    asserts.equals(env, ["-AfooBarValue"], eval_flag_group(fg, st, 128), "testNestedStructureVariableExpansion")

    # testAccessingStructureAsStringFails failure test skipped
    # testAccessingStringValueAsStructureFails failure test skipped
    # testAccessingSequenceAsStructureFails failure test skipped
    # testAccessingMissingStructureFieldFails failure test skipped

    fg = flag_group(iterate_over = "structs", flags = ["-A%{structs.foo}"])
    st = struct(structs = [struct(foo = "foo1Value"), struct(foo = "foo2Value")])
    asserts.equals(env, ["-Afoo1Value", "-Afoo2Value"], eval_flag_group(fg, st, 128), "testSequenceOfStructuresExpansion")

    fg = flag_group(iterate_over = "struct.sequences", flags = ["-A%{struct.sequences.foo}"])
    st = struct(struct = struct(sequences = [struct(foo = "foo1Value"), struct(foo = "foo2Value")]))
    asserts.equals(env, ["-Afoo1Value", "-Afoo2Value"], eval_flag_group(fg, st, 128), "testStructureOfSequencesExpansion")

    fg = flag_group(
        iterate_over = "struct.sequence",
        flag_groups = [
            flag_group(
                iterate_over = "other_sequence",
                flag_groups = [flag_group(flags = ["-A%{struct.sequence} -B%{other_sequence}"])],
            ),
        ],
    )
    st = struct(struct = struct(sequence = ["first", "second"]), other_sequence = ["foo", "bar"])
    asserts.equals(env, ["-Afirst -Bfoo", "-Afirst -Bbar", "-Asecond -Bfoo", "-Asecond -Bbar"], eval_flag_group(fg, st, 128), "testDottedNamesNotAlwaysMeanStructures")

    fg = flag_group(expand_if_available = "struct", flags = ["-A%{struct.foo}", "-B%{struct.bar}"])
    st = struct(struct = struct(foo = "fooValue", bar = "barValue"))
    asserts.equals(env, ["-AfooValue", "-BbarValue"], eval_flag_group(fg, st, 128), "testExpandIfAllAvailableWithStructsExpandsIfPresent")

    fg = flag_group(expand_if_available = "nonexistent", flags = ["-A%{struct.foo}", "-B%{struct.bar}"])
    st = struct(struct = struct(foo = "fooValue", bar = "barValue"))
    asserts.equals(env, [], eval_flag_group(fg, st, 128), "testExpandIfAllAvailableWithStructsDoesntExpandIfMissing")

    fg = flag_group(expand_if_available = "nonexistent", flags = ["-A%{nonexistent.foo}", "-B%{nonexistent.bar}"])
    asserts.equals(env, [], eval_flag_group(fg, struct(), 128), "testExpandIfAllAvailableWithStructsDoesntCrashIfMissing")

    fg = flag_group(expand_if_available = "nonexistent.nonexistant_field", flags = ["-A%{nonexistent.foo}", "-B%{nonexistent.bar}"])
    asserts.equals(env, [], eval_flag_group(fg, struct(), 128), "testExpandIfAllAvailableWithStructFieldDoesntCrashIfMissing")

    fg = flag_group(expand_if_available = "struct.foo", flags = ["-A%{struct.foo}", "-B%{struct.bar}"])
    st = struct(struct = struct(foo = "fooValue", bar = "barValue"))
    asserts.equals(env, ["-AfooValue", "-BbarValue"], eval_flag_group(fg, st, 128), "testExpandIfAllAvailableWithStructFieldExpandsIfPresent")

    fg = flag_group(expand_if_available = "struct.foo", flags = ["-A%{struct.foo}", "-B%{struct.bar}"])
    st = struct(struct = struct(bar = "barValue"))
    asserts.equals(env, [], eval_flag_group(fg, st, 128), "testExpandIfAllAvailableWithStructFieldDoesntExpandIfMissing")

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_available = "struct.foo", flags = ["-A%{struct.foo}"]),
            flag_group(flags = ["-B%{struct.bar}"]),
        ],
    )
    st = struct(struct = struct(bar = "barValue"))
    asserts.equals(env, ["-BbarValue"], eval_flag_group(fg, st, 128), "testExpandIfAllAvailableWithStructFieldScopesRight")

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_not_available = "not_available", flags = ["-foo"]),
            flag_group(expand_if_not_available = "available", flags = ["-bar"]),
        ],
    )
    st = struct(available = "available")
    asserts.equals(env, ["-foo"], eval_flag_group(fg, st, 128), "testExpandIfNoneAvailableExpandsIfNotAvailable")

    # testExpandIfNoneAvailableDoesntExpandIfThereIsOneOfManyAvailable test skipped
    # See https://github.com/bazelbuild/bazel/issues/7008

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_true = "missing", flags = ["-A{missing}"]),
            flag_group(expand_if_false = "missing", flags = ["-B{missing}"]),
        ],
    )
    asserts.equals(env, [], eval_flag_group(fg, struct(), 128), "testExpandIfTrueDoesntExpandIfMissing")

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_true = "struct.bool", flags = ["-A%{struct.foo}", "-B%{struct.bar}"]),
            flag_group(expand_if_false = "struct.bool", flags = ["-X%{struct.foo}", "-Y%{struct.bar}"]),
        ],
    )
    st = struct(struct = struct(bool = 1, foo = "fooValue", bar = "barValue"))
    asserts.equals(env, ["-AfooValue", "-BbarValue"], eval_flag_group(fg, st, 128), "testExpandIfTrueExpandsIfOne")

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_true = "struct.bool", flags = ["-A%{struct.foo}", "-B%{struct.bar}"]),
            flag_group(expand_if_false = "struct.bool", flags = ["-X%{struct.foo}", "-Y%{struct.bar}"]),
        ],
    )
    st = struct(struct = struct(bool = 0, foo = "fooValue", bar = "barValue"))
    asserts.equals(env, ["-XfooValue", "-YbarValue"], eval_flag_group(fg, st, 128), "testExpandIfTrueExpandsIfZero")

    fg = flag_group(
        flag_groups = [
            flag_group(expand_if_equal = variable_with_value(name = "var", value = "equal_value"), flags = ["-foo_%{var}"]),
            flag_group(expand_if_equal = variable_with_value(name = "var", value = "non_equal_value"), flags = ["-bar_%{var}"]),
            flag_group(expand_if_equal = variable_with_value(name = "non_existing_var", value = "non_existing"), flags = ["-baz_%{non_existing_var}"]),
        ],
    )
    st = struct(var = "equal_value")
    asserts.equals(env, ["-foo_equal_value"], eval_flag_group(fg, st, 128), "testExpandIfEqual")

    fg = flag_group(iterate_over = "v", flags = ["%{v}"])
    st = struct(v = ["1", "2"])
    asserts.equals(env, ["1", "2"], eval_flag_group(fg, st, 128), "testListVariableExpansion")

    fg = flag_group(iterate_over = "v1", flags = ["%{v1} %{v2}"])
    st = struct(v1 = ["a1", "a2"], v2 = "b")
    asserts.equals(env, ["a1 b", "a2 b"], eval_flag_group(fg, st, 128), "testListVariableExpansionMixedWithNonListVariable")

    fg = flag_group(iterate_over = "v1", flag_groups = [flag_group(iterate_over = "v2", flags = ["%{v1} %{v2}"])])
    st = struct(v1 = ["a1", "a2"], v2 = ["b1", "b2"])
    asserts.equals(env, ["a1 b1", "a1 b2", "a2 b1", "a2 b2"], eval_flag_group(fg, st, 128), "testNestedListVariableExpansion")

    # testListVariableExpansionMixedWithImplicitlyAccessedListVariableFails failure test skipped

    fg = flag_group(flag_groups = [flag_group(iterate_over = "v", flags = ["-f", "%{v}"]), flag_group(flags = ["-end"])])
    st = struct(v = ["1", "2"])
    asserts.equals(env, ["-f", "1", "-f", "2", "-end"], eval_flag_group(fg, st, 128), "testFlagGroupVariableExpansion 0")

    fg = flag_group(flag_groups = [
        flag_group(iterate_over = "v", flags = ["-f", "%{v}"]),
        flag_group(iterate_over = "v", flags = ["%{v}"]),
    ])
    st = struct(v = ["1", "2"])
    asserts.equals(env, ["-f", "1", "-f", "2", "1", "2"], eval_flag_group(fg, st, 128), "testFlagGroupVariableExpansion 1")

    fg = flag_group(
        iterate_over = "v",
        flag_groups = [flag_group(flags = ["-a"]), flag_group(iterate_over = "v", flags = ["%{v}"]), flag_group(flags = ["-b"])],
    )
    st = struct(v = [["00", "01", "02"], ["10", "11", "12"], ["20", "21", "22"]])
    asserts.equals(env, ["-a", "00", "01", "02", "-b", "-a", "10", "11", "12", "-b", "-a", "20", "21", "22", "-b"], eval_flag_group(fg, st, 128), "testFlagTreeVariableExpansion")

    return unittest.end(env)

eval_flag_group_test = unittest.make(_eval_flag_group_test_impl)

# def toolchain_config_lib_test_suite(name):
#     unittest.suite(
#         name,
#         parse_flag_test,
#         expand_flag_test,
#         eval_flag_group_test,
#     )
