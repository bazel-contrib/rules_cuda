load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts", "unittest")
load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo")
load("//cuda/private:toolchain_config_lib.bzl", "access", "action_config", "config_helper", "create_var_from_value", "env_entry", "env_set", "eval_feature", "eval_flag_group", "exist", "expand_flag", "feature", "feature_set", "flag_group", "flag_set", "parse_flag", "tool", "variable_with_value", "with_feature_set")

def _expect_failure_msg(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, ctx.attr.failure_msg)
    return analysistest.end(env)

failure_msg_test = analysistest.make(
    _expect_failure_msg,
    expect_failure = True,
    attrs = {
        "failure_msg": attr.string(mandatory = True),
    },
)

def _make_utility_failure_test(name, failure_trigger_rule, failure_msg):
    failure_trigger_rule(name = name + "_trigger", tags = ["manual"])
    failure_msg_test(name = name + "_verifier", failure_msg = failure_msg, target_under_test = ":" + name + "_trigger")

def _parse_flag_failure1(ctx):
    f = parse_flag("%")

parse_flag_failure1 = rule(_parse_flag_failure1)
parse_flag_failure1_msg = "expected '{'"

def _parse_flag_failure2(ctx):
    f = parse_flag("%{")

parse_flag_failure2 = rule(_parse_flag_failure2)
parse_flag_failure2_msg = "expected variable name"

def _parse_flag_failure3(ctx):
    f = parse_flag("%{}")

parse_flag_failure3 = rule(_parse_flag_failure3)
parse_flag_failure3_msg = "expected variable name"

def _parse_flag_failure4(ctx):
    f = parse_flag("%{v1.v2 ")

parse_flag_failure4 = rule(_parse_flag_failure4)
parse_flag_failure4_msg = "expected '}'"

def parse_flag_failure_tests():
    _make_utility_failure_test("parse_flag_failure1", parse_flag_failure1, parse_flag_failure1_msg)
    _make_utility_failure_test("parse_flag_failure2", parse_flag_failure2, parse_flag_failure2_msg)
    _make_utility_failure_test("parse_flag_failure3", parse_flag_failure3, parse_flag_failure3_msg)
    _make_utility_failure_test("parse_flag_failure4", parse_flag_failure4, parse_flag_failure4_msg)

def test_parse_flag(env, flag_str, ref_chunks, ref_expandables):
    f = parse_flag(flag_str)
    asserts.equals(env, ref_chunks, f.chunks)
    asserts.equals(env, ref_expandables, f.expandables)

def _parse_flag_test_impl(ctx):
    env = unittest.begin(ctx)
    test_parse_flag(env, "", [], {})

    test_parse_flag(env, "%%", ["%"], {})

    test_parse_flag(env, "%{v}", ["v"], {"v": [0]})

    test_parse_flag(env, " %{v}", [" ", "v"], {"v": [1]})

    test_parse_flag(env, "%%{var}", ["%{var}"], {})

    test_parse_flag(env, " %{v1} %{v2.v3} ", [" ", "v1", " ", "v2.v3", " "], {"v1": [1], "v2.v3": [3]})

    test_parse_flag(env, "%{v1} %{v2.v3}  --flag %{v1}", ["v1", " ", "v2.v3", "  --flag ", "v1"], {"v1": [0, 4], "v2.v3": [2]})

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
    f1.chunks.append("modified")
    f1.expandables["v"].append("modified")
    asserts.equals(env, ["v"], cache[flag_str].chunks)
    asserts.equals(env, {"v": [0]}, cache[flag_str].expandables)
    f2 = parse_flag(flag_str, cache = cache)
    asserts.equals(env, ["v"], f2.chunks)
    asserts.equals(env, {"v": [0]}, f2.expandables)
    asserts.equals(env, ["v"], cache[flag_str].chunks)
    asserts.equals(env, {"v": [0]}, cache[flag_str].expandables)

    return unittest.end(env)

parse_flag_cache_test = unittest.make(_parse_flag_cache_test_impl)

def _var_test_impl(ctx):
    env = unittest.begin(ctx)

    inner_value = struct(inner = "value")
    outer_value = struct(struct = inner_value)
    inner_var = create_var_from_value(inner_value)
    outer_var = create_var_from_value(outer_value)

    asserts.false(env, exist(outer_var, "not_exist"))
    asserts.true(env, exist(outer_var, "struct"))
    asserts.true(env, exist(outer_var, "struct.inner"))

    asserts.equals(env, inner_value, access(outer_var, "struct"))
    asserts.equals(env, "value", access(outer_var, "struct.inner"))

    return unittest.end(env)

var_test = unittest.make(_var_test_impl)

def test_expand_flag(env, flag_info, value, name_to_expand, ref_chunks, ref_expandables):
    var = create_var_from_value(value)
    expand_flag(flag_info, var, name_to_expand)
    asserts.equals(env, ref_chunks, flag_info.chunks)
    asserts.equals(env, ref_expandables, flag_info.expandables)

def _expand_flag_test_impl(ctx):
    env = unittest.begin(ctx)
    test_expand_flag(env, parse_flag("--flag=%{v}"), struct(v = "the-value"), "v", ["--flag=", "the-value"], {})

    test_expand_flag(env, parse_flag("--flag=%{v}"), struct(v = "the-value"), "not_exist", ["--flag=", "v"], {"v": [1]})

    value = struct(v1 = "v1-value", v2 = "v2-value")
    flag_info = parse_flag("--flag=%{v1}x%{v2}")
    test_expand_flag(env, flag_info, value, "v2", ["--flag=", "v1", "x", "v2-value"], {"v1": [1]})
    test_expand_flag(env, flag_info, value, "v1", ["--flag=", "v1-value", "x", "v2-value"], {})

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

def _test_accessing_structure_as_string_fails(ctx):
    fg = flag_group(flags = ["-A%{struct}"])
    st = struct(struct = struct(foo = "fooValue", bar = "barValue"))
    eval_flag_group(fg, st, 128)

test_accessing_structure_as_string_fails = rule(_test_accessing_structure_as_string_fails)
test_accessing_structure_as_string_fails_msg = "Cannot expand variable 'struct': expected string, found struct"  # testAccessingStructureAsStringFails

def _test_accessing_string_value_as_structure_fails(ctx):
    fg = flag_group(flags = ["-A%{stringVar.foo}"])
    st = struct(stringVar = "stringVarValue")
    eval_flag_group(fg, st, 128)

test_accessing_string_value_as_structure_fails = rule(_test_accessing_string_value_as_structure_fails)
test_accessing_string_value_as_structure_fails_msg = "Cannot expand variable 'stringVar.foo': variable 'stringVar' is string, expected structure"  # testAccessingStringValueAsStructureFails

def _test_accessing_sequence_as_structure_fails(ctx):
    fg = flag_group(flags = ["-A%{sequence.foo}"])
    st = struct(sequence = ["foo1", "foo2"])
    eval_flag_group(fg, st, 128)

test_accessing_sequence_as_structure_fails = rule(_test_accessing_sequence_as_structure_fails)
test_accessing_sequence_as_structure_fails_msg = "Cannot expand variable 'sequence.foo': variable 'sequence' is sequence, expected structure"  # testAccessingSequenceAsStructureFails

def _test_accessing_missing_structure_field_fails(ctx):
    fg = flag_group(flags = ["-A%{struct.missing}"])
    st = struct(struct = struct(bar = "barValue"))
    eval_flag_group(fg, st, 128)

test_accessing_missing_structure_field_fails = rule(_test_accessing_missing_structure_field_fails)
test_accessing_missing_structure_field_fails_msg = "Cannot expand variable 'struct.missing'"  # testAccessingMissingStructureFieldFails

# testExpandIfNoneAvailableDoesntExpandIfThereIsOneOfManyAvailable test skipped, See https://github.com/bazelbuild/bazel/issues/7008

def _test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable(ctx):
    fg = flag_group(
        iterate_over = "v1",
        flags = ["%{v1} %{v2}"],
    )
    st = struct(v1 = ["a1", "a2"], v2 = ["b1", "b2"])
    eval_flag_group(fg, st, 128)

test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable = rule(_test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable)
test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable_msg = "Cannot expand variable 'v2': expected string, found ["  # testListVariableExpansionMixedWithImplicitlyAccessedListVariableFails

def eval_flag_group_failure_tests():
    _make_utility_failure_test(
        "test_accessing_structure_as_string_fails",
        test_accessing_structure_as_string_fails,
        test_accessing_structure_as_string_fails_msg,
    )
    _make_utility_failure_test(
        "test_accessing_string_value_as_structure_fails",
        test_accessing_string_value_as_structure_fails,
        test_accessing_string_value_as_structure_fails_msg,
    )
    _make_utility_failure_test(
        "test_accessing_sequence_as_structure_fails",
        test_accessing_sequence_as_structure_fails,
        test_accessing_sequence_as_structure_fails_msg,
    )
    _make_utility_failure_test(
        "test_accessing_missing_structure_field_fails",
        test_accessing_missing_structure_field_fails,
        test_accessing_missing_structure_field_fails_msg,
    )
    _make_utility_failure_test(
        "test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable",
        test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable,
        test_list_variable_expansion_mixed_with_implicitly_accessed_list_variable_msg,
    )

def get_enabled_selectables(selectables = None, info = None, requested = None):
    info = config_helper.configure_features(selectables = selectables, selectables_info = info, requested_features = requested)
    return sorted([k for k, v in info.enabled.items() if v == True])

def _feature_constraint_test_impl(ctx):
    env = unittest.begin(ctx)

    features = [
        feature(name = "a", implies = ["b", "c"]),
        feature(name = "b"),
        feature(name = "c", implies = ["d"]),
        feature(name = "d"),
        feature(name = "e"),
    ]
    asserts.equals(env, ["a", "b", "c", "d"], get_enabled_selectables(features, requested = ["a"]), "testImplies")

    features = [
        feature(name = "a", requires = [feature_set(features = ["b"])]),
        feature(name = "b", requires = [feature_set(features = ["c"])]),
        feature(name = "c"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a"]), "testRequires 0")
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a", "b"]), "testRequires 1")
    asserts.equals(env, ["c"], get_enabled_selectables(features, requested = ["a", "c"]), "testRequires 2")
    asserts.equals(env, ["a", "b", "c"], get_enabled_selectables(features, requested = ["a", "b", "c"]), "testRequires 3")

    features = [
        feature(name = "a"),
        feature(name = "b", implies = ["a"], requires = [feature_set(features = ["c"])]),
        feature(name = "c"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["b"]), "testDisabledRequirementChain 0")
    features = [
        feature(name = "a"),
        feature(name = "b", implies = ["a"], requires = [feature_set(features = ["c"])]),
        feature(name = "c"),
        feature(name = "d", implies = ["e"], requires = [feature_set(features = ["c"])]),
        feature(name = "e"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["b", "d"]), "testDisabledRequirementChain 1")

    features = [
        feature(name = "0", implies = ["a"]),
        feature(name = "a"),
        feature(name = "b", implies = ["c"], requires = [feature_set(features = ["a"])]),
        feature(name = "c"),
        feature(name = "d", implies = ["e"], requires = [feature_set(features = ["c"])]),
        feature(name = "e"),
    ]
    asserts.equals(env, ["0", "a", "b", "c", "d", "e"], get_enabled_selectables(features, requested = ["0", "b", "d"]), "testEnabledRequirementChain")

    features = [
        feature(name = "a", requires = [feature_set(features = ["b", "c"]), feature_set(features = ["d"])]),
        feature(name = "b"),
        feature(name = "c"),
        feature(name = "d"),
    ]
    asserts.equals(env, ["a", "b", "c"], get_enabled_selectables(features, requested = ["a", "b", "c"]), "testLogicInRequirements 0")
    asserts.equals(env, ["b"], get_enabled_selectables(features, requested = ["a", "b"]), "testLogicInRequirements 1")
    asserts.equals(env, ["c"], get_enabled_selectables(features, requested = ["a", "c"]), "testLogicInRequirements 2")
    asserts.equals(env, ["a", "d"], get_enabled_selectables(features, requested = ["a", "d"]), "testLogicInRequirements 3")

    features = [
        feature(name = "a", implies = ["b"]),
        feature(name = "b", requires = [feature_set(features = ["c"])]),
        feature(name = "c"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a"]), "testImpliesImpliesRequires")

    features = [
        feature(name = "a", implies = ["b", "c", "d"]),
        feature(name = "b"),
        feature(name = "c", requires = [feature_set(features = ["e"])]),
        feature(name = "d"),
        feature(name = "e"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a"]), "testMultipleImplies 0")
    asserts.equals(env, ["a", "b", "c", "d", "e"], get_enabled_selectables(features, requested = ["a", "e"]), "testMultipleImplies 1")

    features = [
        feature(name = "a", implies = ["b"], requires = [feature_set(features = ["c"])]),
        feature(name = "b"),
        feature(name = "c"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a"]), "testDisabledFeaturesDoNotEnableImplications")

    features = [
        feature(name = "a", implies = ["b"]),
        feature(name = "b", implies = ["a"]),
    ]
    asserts.equals(env, ["a", "b"], get_enabled_selectables(features, requested = ["a"]), "testImpliesWithCycle 0")
    asserts.equals(env, ["a", "b"], get_enabled_selectables(features, requested = ["b"]), "testImpliesWithCycle 1")

    features = [
        feature(name = "a", implies = ["b", "c", "d"]),
        feature(name = "b"),
        feature(name = "c", requires = [feature_set(features = ["e"])]),
        feature(name = "d", requires = [feature_set(features = ["f"])]),
        feature(name = "e", requires = [feature_set(features = ["c"])]),
        feature(name = "f"),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["a", "e"]), "testMultipleImpliesCycle 0")
    asserts.equals(env, ["a", "b", "c", "d", "e", "f"], get_enabled_selectables(features, requested = ["a", "e", "f"]), "testMultipleImpliesCycle 1")

    features = [
        feature(name = "a", requires = [feature_set(features = ["b"])]),
        feature(name = "b", requires = [feature_set(features = ["a"])]),
        feature(name = "c", implies = ["a"]),
        feature(name = "d", implies = ["b"]),
    ]
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["c"]), "testRequiresWithCycle 0")
    asserts.equals(env, [], get_enabled_selectables(features, requested = ["d"]), "testRequiresWithCycle 1")
    asserts.equals(env, ["a", "b", "c", "d"], get_enabled_selectables(features, requested = ["c", "d"]), "testRequiresWithCycle 2")

    features = [
        feature(name = "a"),
        feature(name = "b", implies = ["d"], requires = [feature_set(features = ["a"])]),
        feature(name = "c", implies = ["d"]),
        feature(name = "d"),
    ]
    asserts.equals(env, ["c", "d"], get_enabled_selectables(features, requested = ["b", "c"]), "testImpliedByOneEnabledAndOneDisabledFeature")

    features = [
        feature(name = "a", requires = [feature_set(features = ["b"]), feature_set(features = ["c"])]),
        feature(name = "b"),
        feature(name = "c", requires = [feature_set(features = ["d"])]),
        feature(name = "d"),
    ]
    asserts.equals(env, ["a", "b"], get_enabled_selectables(features, requested = ["a", "b", "c"]), "testRequiresOneEnabledAndOneUnsupportedFeature")

    return unittest.end(env)

feature_constraint_test = unittest.make(_feature_constraint_test_impl)

def _feature_flag_sets_test_impl(ctx):
    env = unittest.begin(ctx)

    feat = feature(
        name = "a",
        flag_sets = [
            flag_set(actions = ["c"], flag_groups = [flag_group(expand_if_available = "v", flags = ["%{v}"])]),
            flag_set(actions = ["c"], flag_groups = [flag_group(flags = ["unconditional"])]),
        ],
    )
    asserts.equals(env, ["unconditional"], eval_feature(feat, struct(), "c", None), "testFlagGroupsWithMissingVariableIsNotExpanded")

    # NOTE: expand_if_all_available is deprecated, use nested flag_group to express the same logic
    feat = feature(
        name = "a",
        flag_sets = [
            flag_set(actions = ["c"], flag_groups = [flag_group(expand_if_available = "v", flags = ["%{v}"])]),
            flag_set(actions = ["c"], flag_groups = [
                flag_group(
                    expand_if_available = "v",
                    flag_groups = [flag_group(expand_if_available = "w", flags = ["%{v}%{w}"])],
                ),
            ]),
            flag_set(actions = ["c"], flag_groups = [flag_group(flags = ["unconditional"])]),
        ],
    )
    asserts.equals(env, ["1", "unconditional"], eval_feature(feat, struct(v = "1"), "c", None), "testOnlyFlagGroupsWithAllVariablesPresentAreExpanded")

    feat = feature(
        name = "a",
        flag_sets = [
            flag_set(actions = ["c"], flag_groups = [flag_group(expand_if_available = "v", iterate_over = "v", flags = ["%{v}"])]),
            flag_set(actions = ["c"], flag_groups = [
                flag_group(
                    expand_if_available = "v",
                    iterate_over = "v",
                    flag_groups = [flag_group(expand_if_available = "w", flags = ["%{v}%{w}"])],
                ),
            ]),
            flag_set(actions = ["c"], flag_groups = [flag_group(flags = ["unconditional"])]),
        ],
    )
    asserts.equals(env, ["1", "2", "unconditional"], eval_feature(feat, struct(v = ["1", "2"]), "c", None), "testOnlyInnerFlagGroupIsIteratedWithSequenceVariable")
    asserts.equals(env, ["1", "2", "13", "23", "unconditional"], eval_feature(feat, struct(v = ["1", "2"], w = "3"), "c", None), "testFlagSetsAreIteratedIndividuallyForSequenceVariables")

    return unittest.end(env)

feature_flag_sets_test = unittest.make(_feature_flag_sets_test_impl)

def create_toolchain_config(action_configs = [], features = [], artifact_name_patterns = [], toolchain_identifier = "nvcc", cuda_path = None):
    return CudaToolchainConfigInfo(
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        features = features,
        toolchain_identifier = toolchain_identifier,
        cuda_path = cuda_path,
    )

def create_config_info(features, requested = [], unsupported = []):
    return config_helper.configure_features(selectables = features, requested_features = requested, unsupported_features = unsupported)

def _feature_configuration_test_impl(ctx):
    env = unittest.begin(ctx)

    config_info = create_config_info(
        [
            feature(name = "a", flag_sets = [flag_set(actions = ["c"], flag_groups = [flag_group(flags = ["-f", "%{v}"])])]),
            feature(name = "b", implies = ["a"]),
        ],
        ["b"],
    )
    asserts.equals(env, ["a", "b"], config_helper.get_enabled_feature(config_info), "testConfiguration get_enabled_feature")
    asserts.equals(env, ["-f", "1"], config_helper.get_command_line(config_info, "c", struct(v = "1")), "testConfiguration get_enabled_feature")

    config_info = create_config_info([feature(name = "a"), feature(name = "b", enabled = True)])
    asserts.equals(env, ["b"], config_helper.get_default_features_and_action_configs(config_info), "testDefaultFeatures")

    config_info = create_config_info([action_config(action_name = "a"), action_config(action_name = "b", enabled = True)])
    asserts.equals(env, ["b"], config_helper.get_default_features_and_action_configs(config_info), "testDefaultActionConfigs")

    config_info = create_config_info(
        [
            feature(
                name = "a",
                flag_sets = [flag_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(features = ["b"])],
                    flag_groups = [flag_group(flags = ["dummy_flag"])],
                )],
            ),
            feature(name = "b"),
        ],
        ["a", "b"],
    )
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_oneSetOneFeature")

    config_info = create_config_info(
        [
            feature(
                name = "a",
                flag_sets = [flag_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(features = ["b", "c"])],
                    flag_groups = [flag_group(flags = ["dummy_flag"])],
                )],
            ),
            feature(name = "b"),
            feature(name = "c"),
        ],
        ["a", "b", "c"],
    )
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_oneSetMultipleFeatures")

    features = [
        feature(
            name = "a",
            flag_sets = [flag_set(
                actions = ["c++-compile"],
                with_features = [
                    with_feature_set(features = ["b1", "c1"]),
                    with_feature_set(features = ["b2", "c2"]),
                ],
                flag_groups = [flag_group(flags = ["dummy_flag"])],
            )],
        ),
        feature(name = "b1"),
        feature(name = "c1"),
        feature(name = "b2"),
        feature(name = "c2"),
    ]
    config_info = create_config_info(features, ["a", "b1", "c1", "b2", "c2"])
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_mulipleSetsMultipleFeatures 0")
    config_info = create_config_info(features, ["a", "b1", "c1"])
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_mulipleSetsMultipleFeatures 1")
    config_info = create_config_info(features, ["a", "b1", "b2"])
    asserts.equals(env, [], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_mulipleSetsMultipleFeatures 2")

    features = [
        feature(
            name = "a",
            flag_sets = [flag_set(
                actions = ["c++-compile"],
                with_features = [
                    with_feature_set(not_features = ["x", "y"], features = ["z"]),
                    with_feature_set(not_features = ["q"]),
                ],
                flag_groups = [flag_group(flags = ["dummy_flag"])],
            )],
        ),
        feature(name = "x"),
        feature(name = "y"),
        feature(name = "z"),
        feature(name = "q"),
    ]
    config_info = create_config_info(features, ["a"])
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_notFeature 0")
    config_info = create_config_info(features, ["a", "q"])
    asserts.equals(env, [], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_notFeature 1")
    config_info = create_config_info(features, ["a", "q", "z"])
    asserts.equals(env, ["dummy_flag"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_notFeature 2")
    config_info = create_config_info(features, ["a", "q", "x", "z"])
    asserts.equals(env, [], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_notFeature 3")
    config_info = create_config_info(features, ["a", "q", "x", "y", "z"])
    asserts.equals(env, [], config_helper.get_command_line(config_info, "c++-compile", struct()), "testWithFeature_notFeature 4")

    features = [
        action_config(
            action_name = "action-a",
            tools = [tool(path = "toolchain/feature-a", with_features = [with_feature_set(features = ["feature-a"])])],
        ),
        feature(
            name = "activates-action-a",
            implies = ["action-a"],
        ),
    ]
    config_info = create_config_info(features, ["activates-action-a"])
    asserts.true(env, config_helper.action_is_enabled(config_info, "action-a"), "testActivateActionConfigFromFeature")

    features = [
        action_config(
            action_name = "action-a",
            tools = [tool(path = "toolchain/feature-a", with_features = [with_feature_set(features = ["feature-a"])])],
        ),
        feature(
            name = "requires-action-a",
            requires = [feature_set(features = ["action-a"])],
        ),
    ]
    config_info = create_config_info(features, ["requires-action-a"])
    asserts.false(env, config_helper.is_enabled(config_info, "requires-action-a"), "testFeatureCanRequireActionConfig 0")
    config_info = create_config_info(features, ["action-a", "requires-action-a"])
    asserts.true(env, config_helper.is_enabled(config_info, "requires-action-a"), "testFeatureCanRequireActionConfig 1")

    features = [
        action_config(
            action_name = "action-a",
            tools = [tool(path = "toolchain/a")],
        ),
        feature(
            name = "activates-action-a",
            implies = ["action-a"],
        ),
    ]
    config_info = create_config_info(features, ["activates-action-a"])
    asserts.equals(env, "toolchain/a", config_helper.get_tool_for_action(config_info, "action-a"), "testSimpleActionTool")

    features = [
        action_config(
            action_name = "action-a",
            tools = [
                tool(
                    path = "toolchain/feature-a-and-b",
                    with_features = [with_feature_set(features = ["feature-a", "feature-b"])],
                ),
                tool(
                    path = "toolchain/feature-a-and-not-c",
                    with_features = [with_feature_set(features = ["feature-a"], not_features = ["feature-c"])],
                ),
                tool(
                    path = "toolchain/feature-b-or-c",
                    with_features = [
                        with_feature_set(features = ["feature-b"]),
                        with_feature_set(features = ["feature-c"]),
                    ],
                ),
                tool(path = "toolchain/default"),
            ],
        ),
        feature(name = "feature-a"),
        feature(name = "feature-b"),
        feature(name = "feature-c"),
        feature(name = "activates-action-a", implies = ["action-a"]),
    ]
    config_info = create_config_info(features, ["feature-a", "activates-action-a"])
    asserts.equals(env, "toolchain/feature-a-and-not-c", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 0")

    config_info = create_config_info(features, ["feature-a", "feature-c", "activates-action-a"])
    asserts.equals(env, "toolchain/feature-b-or-c", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 1")

    config_info = create_config_info(features, ["feature-b", "activates-action-a"])
    asserts.equals(env, "toolchain/feature-b-or-c", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 2")

    config_info = create_config_info(features, ["feature-c", "activates-action-a"])
    asserts.equals(env, "toolchain/feature-b-or-c", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 3")

    config_info = create_config_info(features, ["feature-a", "feature-b", "activates-action-a"])
    asserts.equals(env, "toolchain/feature-a-and-b", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 4")

    config_info = create_config_info(features, ["activates-action-a"])
    asserts.equals(env, "toolchain/default", config_helper.get_tool_for_action(config_info, "action-a"), "testActionToolFromFeatureSet 5")

    features = [
        action_config(
            action_name = "action-a",
            tools = [tool(path = "toolchain/feature-a", with_features = [with_feature_set(features = ["feature-a"])])],
        ),
    ]
    config_info = create_config_info(features, ["action-a"])
    asserts.true(env, config_helper.action_is_enabled(config_info, "action-a"), "testActivateActionConfigDirectly")

    features = [
        action_config(
            action_name = "action-a",
            tools = [tool(path = "toolchain/feature-a", with_features = [with_feature_set(features = ["feature-a"])])],
            implies = ["activated-feature"],
        ),
        feature(name = "activated-feature"),
    ]
    config_info = create_config_info(features, ["action-a"])
    asserts.true(env, config_helper.action_is_enabled(config_info, "activated-feature"), "testActionConfigCanActivateFeature")

    features = [
        action_config(
            action_name = "c++-compile",
            flag_sets = [flag_set(flag_groups = [flag_group(flags = ["foo"])])],
        ),
    ]
    config_info = create_config_info(features, ["c++-compile"])
    asserts.equals(env, ["foo"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testFlagsFromActionConfig")

    return unittest.end(env)

feature_configuration_test = unittest.make(_feature_configuration_test_impl)

def _test_feature_name_collision(ctx):
    create_config_info([feature(name = "<<<collision>>>"), feature(name = "<<<collision>>>")])

test_feature_name_collision = rule(_test_feature_name_collision)
test_feature_name_collision_msg = "<<<collision>>>"  # testFeatureNameCollision

def _test_reference_to_undefined_feature(ctx):
    implies = ["<<<undefined>>>"]
    create_config_info([feature(name = "a", implies = implies)])

test_reference_to_undefined_feature = rule(_test_reference_to_undefined_feature)
test_reference_to_undefined_feature_msg = "<<<undefined>>> is not defined"  # testReferenceToUndefinedFeature

def _test_error_for_no_matching_tool(ctx):
    config_info = create_config_info([
        action_config(
            action_name = "action-a",
            tools = [tool(
                path = "toolchain/feature-a",
                with_features = [with_feature_set(features = ["feature-a"])],
            )],
        ),
        feature(name = "feature-a"),
        feature(name = "activates-action-a", implies = ["action-a"]),
    ])
    config_helper.get_tool_for_action(config_info, "action-a")

test_error_for_no_matching_tool = rule(_test_error_for_no_matching_tool)
test_error_for_no_matching_tool_msg = "Matching tool for action action-a not found for given feature configuration"  # testErrorForNoMatchingTool

def _test_invalid_action_configuration_duplicate_action_configs(ctx):
    create_config_info([action_config(action_name = "action-a"), action_config(action_name = "action-a")])

test_invalid_action_configuration_duplicate_action_configs = rule(_test_invalid_action_configuration_duplicate_action_configs)
test_invalid_action_configuration_duplicate_action_configs_msg = "feature or action config 'action-a' was specified multiple times."  # testInvalidActionConfigurationDuplicateActionConfigs

# testInvalidActionConfigurationMultipleActionConfigsForAction failure test skipped, because config_name in action_config has gone!

def _error_for_flag_from_action_config_with_specified_action(ctx):
    features = [
        action_config(
            action_name = "c++-compile",
            flag_sets = [flag_set(
                actions = ["c++-compile"],
                flag_groups = [flag_group(flags = ["foo"])],
            )],
        ),
    ]
    config_info = create_config_info(features, ["c++-compile"])

error_for_flag_from_action_config_with_specified_action = rule(_error_for_flag_from_action_config_with_specified_action)
error_for_flag_from_action_config_with_specified_action_msg = (
    "action_config {} specifies actions.  An action_config's flag sets automatically apply to the " +
    "configured action.  Thus, you must not specify action lists in an action_config's flag set."
).format("c++-compile")  # testErrorForFlagFromActionConfigWithSpecifiedAction

def _feature_configuration_provides_collision(ctx):
    features = [
        feature(name = "a", provides = ["provides_string"]),
        feature(name = "b", provides = ["provides_string"]),
    ]
    config_info = create_config_info(features, ["a", "b"])

feature_configuration_provides_collision = rule(_feature_configuration_provides_collision)
feature_configuration_provides_collision_msg = "a b"  # testProvidesCollision

def feature_configuration_failure_tests():
    _make_utility_failure_test(
        "test_feature_name_collision",
        test_feature_name_collision,
        test_feature_name_collision_msg,
    )
    _make_utility_failure_test(
        "test_reference_to_undefined_feature",
        test_reference_to_undefined_feature,
        test_reference_to_undefined_feature_msg,
    )
    _make_utility_failure_test(
        "test_error_for_no_matching_tool",
        test_error_for_no_matching_tool,
        test_error_for_no_matching_tool_msg,
    )
    _make_utility_failure_test(
        "test_invalid_action_configuration_duplicate_action_configs",
        test_invalid_action_configuration_duplicate_action_configs,
        test_invalid_action_configuration_duplicate_action_configs_msg,
    )
    _make_utility_failure_test(
        "error_for_flag_from_action_config_with_specified_action",
        error_for_flag_from_action_config_with_specified_action,
        error_for_flag_from_action_config_with_specified_action_msg,
    )
    _make_utility_failure_test(
        "feature_configuration_provides_collision",
        feature_configuration_provides_collision,
        feature_configuration_provides_collision_msg,
    )

def _feature_configuration_flags_order_test_impl(ctx):
    env = unittest.begin(ctx)

    config_info = create_config_info(
        [
            feature(
                name = "a",
                flag_sets = [
                    flag_set(actions = ["c++-compile"], flag_groups = [flag_group(flags = ["-a-c++-compile"])]),
                    flag_set(actions = ["link"], flag_groups = [flag_group(flags = ["-a-c++-compile"])]),
                ],
            ),
            feature(
                name = "b",
                flag_sets = [
                    flag_set(actions = ["c++-compile"], flag_groups = [flag_group(flags = ["-b-c++-compile"])]),
                    flag_set(actions = ["link"], flag_groups = [flag_group(flags = ["-b-link"])]),
                ],
            ),
        ],
        ["a", "b"],
    )
    asserts.equals(env, ["-a-c++-compile", "-b-c++-compile"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testFlagOrderEqualsSpecOrder")

    # test by me, flag order should not be affect by enable order
    config_info = create_config_info(
        [
            feature(
                name = "a",
                flag_sets = [
                    flag_set(actions = ["c++-compile"], flag_groups = [flag_group(flags = ["-a-c++-compile"])]),
                    flag_set(actions = ["link"], flag_groups = [flag_group(flags = ["-a-c++-compile"])]),
                ],
            ),
            feature(
                name = "b",
                enabled = True,
                flag_sets = [
                    flag_set(actions = ["c++-compile"], flag_groups = [flag_group(flags = ["-b-c++-compile"])]),
                    flag_set(actions = ["link"], flag_groups = [flag_group(flags = ["-b-link"])]),
                ],
            ),
        ],
        ["a"],
    )
    asserts.equals(env, ["-a-c++-compile", "-b-c++-compile"], config_helper.get_command_line(config_info, "c++-compile", struct()), "testFlagOrderEqualsSpecOrderNotEnableOrder (my test)")

    return unittest.end(env)

feature_configuration_flags_order_test = unittest.make(_feature_configuration_flags_order_test_impl)

def _feature_configuration_env_test_impl(ctx):
    env = unittest.begin(ctx)

    config_info = create_config_info([
        feature(
            name = "a",
            env_sets = [env_set(
                actions = ["c++-compile"],
                env_entries = [env_entry(key = "foo", value = "bar"), env_entry(key = "cat", value = "meow")],
            )],
            flag_sets = [flag_set(
                actions = ["c++-compile"],
                flag_groups = [flag_group(flags = ["-a-c++-compile"])],
            )],
        ),
        feature(
            name = "b",
            env_sets = [
                env_set(
                    actions = ["c++-compile"],
                    env_entries = [env_entry(key = "dog", value = "woof")],
                ),
                env_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(features = ["d"])],
                    env_entries = [env_entry(key = "withFeature", value = "value1")],
                ),
                env_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(features = ["e"])],
                    env_entries = [env_entry(key = "withoutFeature", value = "value2")],
                ),
                env_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(not_features = ["f"])],
                    env_entries = [env_entry(key = "withNotFeature", value = "value3")],
                ),
                env_set(
                    actions = ["c++-compile"],
                    with_features = [with_feature_set(not_features = ["g"])],
                    env_entries = [env_entry(key = "withoutNotFeature", value = "value4")],
                ),
            ],
        ),
        feature(
            name = "c",
            env_sets = [env_set(
                actions = ["c++-compile"],
                env_entries = [env_entry(key = "doNotInclude", value = "doNotIncludePlease")],
            )],
        ),
        feature(name = "d"),
        feature(name = "e"),
        feature(name = "f"),
        feature(name = "g"),
    ], ["a", "b", "d", "f"])
    environ = config_helper.get_environment_variables(config_info, "c++-compile", struct())
    ref_environ = {"foo": "bar", "cat": "meow", "dog": "woof", "withFeature": "value1", "withoutNotFeature": "value4"}
    asserts.equals(env, ref_environ, environ, "testEnvVars")
    asserts.equals(env, ref_environ.keys(), environ.keys(), "testEnvVars dict in order")

    return unittest.end(env)

feature_configuration_env_test = unittest.make(_feature_configuration_env_test_impl)

def _feature_configuration_test_unsuppoted_features_impl(ctx):
    env = unittest.begin(ctx)

    features = [
        feature(name = "a", implies = ["b", "c", "d"]),
        feature(name = "b"),
        feature(name = "c", requires = [feature_set(features = ["e"])]),
        feature(name = "d", requires = [feature_set(features = ["f"])]),
        feature(name = "e", requires = [feature_set(features = ["c"])]),
        feature(name = "f"),
    ]
    config_info = create_config_info(features, requested = ["a", "b", "c", "d", "e", "f"], unsupported = ["a"])
    asserts.equals(env, ["b", "c", "d", "e", "f"], config_helper.get_enabled_feature(config_info), "configure_features with unsupported_features 0")

    config_info = create_config_info(features, requested = ["a", "b", "c", "d", "e", "f"], unsupported = ["b"])
    asserts.equals(env, ["c", "d", "e", "f"], config_helper.get_enabled_feature(config_info), "configure_features with unsupported_features 1")

    config_info = create_config_info(features, requested = ["a", "b", "c", "d", "e", "f"], unsupported = ["f"])
    asserts.equals(env, ["b", "c", "e"], config_helper.get_enabled_feature(config_info), "configure_features with unsupported_features 3")

    return unittest.end(env)

feature_configuration_unsuppoted_features_test = unittest.make(_feature_configuration_test_unsuppoted_features_impl)
