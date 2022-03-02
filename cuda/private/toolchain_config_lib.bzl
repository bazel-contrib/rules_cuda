load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    _feature = "feature",
    _flag_group = "flag_group",
    _flag_set = "flag_set",
    _tool = "tool",
    _tool_path = "tool_path",
    _variable_with_value = "variable_with_value",
)

_MAX_FLAG_LEN = 1024

def _tok_var(chars):
    if len(chars) == 0 or chars[-1] != "{":
        fail("expected '{'")
    chars.pop()
    var = []
    if len(chars) and (chars[-1].isalnum() or chars[-1] == "_"):
        var.append(chars.pop())
    else:
        fail("expected variable name")
    for i in range(_MAX_FLAG_LEN + 1):
        if len(chars) and chars[-1].isalnum() or chars[-1] in "._":
            var.append(chars.pop())
        else:
            break
    if len(chars) and chars[-1] == "}":
        chars.pop()
    else:
        fail("expected '}'")
    return "".join(var)

_FlagInfo = provider(
    "",
    fields = {
        "chunks": "",
        "expandables": "",
    },
)

def _copy_flag_info(flag_info):
    expandables = {}
    for k, v in flag_info.expandables.items():
        expandables[k] = v[:]
    return _FlagInfo(
        chunks = flag_info.chunks[:],
        expandables = expandables,
    )

def parse_flag(raw_flag, cache = None):
    if len(raw_flag) > _MAX_FLAG_LEN:
        fail(raw_flag, "is too long!")
    if cache != None and raw_flag in cache:
        return _copy_flag_info(cache[raw_flag])
    curr = None
    chars = reversed(list(raw_flag.elems()))
    result = []
    expandable_indices = []
    for i in range(_MAX_FLAG_LEN + 1):
        if len(chars) == 0:
            break
        if curr == None:
            curr = chars.pop()
        if curr == "%":
            if len(chars) and chars[-1] == "%":
                result.append(chars.pop())
            else:
                expandable_indices.append(len(result))
                result.append(_tok_var(chars))
            curr = None
            continue
        result.append(curr)
        curr = None

    compact_result = []
    compact_expandable_indices = []
    expandables = {}

    tmp = []
    for i, r in enumerate(result):
        if i in expandable_indices:
            if len(tmp):
                compact_result.append("".join(tmp))
            compact_expandable_indices.append(len(compact_result))
            compact_result.append(r)
            tmp = []
        else:
            tmp.append(r)
    if len(tmp):
        compact_result.append("".join(tmp))

    for i in compact_expandable_indices:
        expandables.setdefault(compact_result[i], [])
    for i in compact_expandable_indices:
        expandables[compact_result[i]].append(i)

    flag_info = _FlagInfo(
        chunks = compact_result,
        expandables = expandables,
    )
    if cache != None:
        cache[raw_flag] = _copy_flag_info(flag_info)
    return flag_info

_NestingVarInfo = provider(
    """""",
    fields = {
        "parent": "",
        "this": "",
    },
)

_VAR_NESTING_MAX_DEPTH = 64

def _single_access(value, path_list, ret):
    v = value
    for i, name in enumerate(path_list):
        if hasattr(v, name):
            v = getattr(v, name)
        else:
            return False
    ret.append(v)
    return True

def exist(input_var, path = None, path_list = None):
    if path_list == None:
        path_list = path.split(".")
    var = None
    parent_nesting_var = input_var
    for _ in range(_VAR_NESTING_MAX_DEPTH):
        if parent_nesting_var == None:
            break
        var = parent_nesting_var.this
        parent_nesting_var = parent_nesting_var.parent
        if _single_access(var, path_list, []):
            return True
    return False

def access(var, path = None, path_list = None, fail_if_not_available = True):
    if path_list == None:
        path_list = path.split(".")
    ret = []
    value = None
    parent_nesting_var = var
    for _ in range(_VAR_NESTING_MAX_DEPTH):
        if parent_nesting_var == None:
            break
        value = parent_nesting_var.this
        parent_nesting_var = parent_nesting_var.parent
        if _single_access(value, path_list, ret):
            return ret[0]
    if fail_if_not_available:
        fail("Cannot access", ".".join(path_list))
    else:
        return None

def create_var_from_value(value, parent = None, path = None, path_list = None):
    if path == None and path_list == None:
        return _NestingVarInfo(this = value, parent = parent)
    if path_list == None:
        path_list = path.split(".")
    v = value
    for i in range(len(path_list) - 1, -1, -1):
        name = path_list[i]
        v = struct(**{name: v})
    return _NestingVarInfo(this = v, parent = parent)

def expand_flag(flag_info, var, name):
    if len(flag_info.expandables) == 0 or name not in flag_info.expandables:
        return
    if not exist(var, name):
        return
    value = access(var, name)
    if type(value) != "string":
        fail("Cannot expand variable '" + name + "': expected string, found", value)
    for i in flag_info.expandables[name]:
        flag_info.chunks[i] = value
    flag_info.expandables.pop(name)

def _can_be_expanded(fg, var):
    if fg.expand_if_available != None and not exist(var, fg.expand_if_available):
        return False
    if fg.expand_if_not_available != None and exist(var, fg.expand_if_not_available):
        return False
    if fg.expand_if_true != None and access(var, fg.expand_if_true, fail_if_not_available = False) not in [True, 1]:
        return False
    if fg.expand_if_false != None and access(var, fg.expand_if_false, fail_if_not_available = False) not in [False, 0]:
        return False
    if fg.expand_if_equal != None and (not exist(var, fg.expand_if_equal.name) or access(var, fg.expand_if_equal.name) != fg.expand_if_equal.value):
        return False
    return True

def _expand_flag_infos_in_current_scope(flag_infos, var):
    for flag_info in flag_infos:
        for name in flag_info.expandables.keys():
            expand_flag(flag_info, var, name)

def _eval_flags_or_flag_groups(stack, ret, fg, var, recursion_depth, parse_flag_cache):
    if len(fg.flags) > 0 and len(fg.flag_groups) == 0:
        # no need to reverse, because it is not push stack
        flag_infos = [parse_flag(flag_str, parse_flag_cache) for flag_str in fg.flags]
        _expand_flag_infos_in_current_scope(flag_infos, var)
        ret[-1].extend(flag_infos)
    elif len(fg.flags) == 0 and len(fg.flag_groups) > 0:
        # reverse push stack, so that we can maintain in-order transverse
        for i in range(len(fg.flag_groups) - 1, -1, -1):
            stack.append([fg.flag_groups[i], var, recursion_depth + 1, False])
    else:
        fail(fg, "is invalid, either flags or flag_groups must be specified.")

def _eval_flag_group_impl(stack, ret, fg, var, eval_iterations):
    parse_flag_cache = {}
    stack.append([fg, var, 1, False])
    recursion_depth = 0
    for _ in range(eval_iterations):
        if len(stack) == 0:
            break
        fg, var, recursion_depth, entered = stack[-1]
        if entered:  # return from a recursive call. We need to handle the returned value.
            # Since we are returning from another function, the variable socpe is different,
            # we need to expand all flags in current scope again.
            _expand_flag_infos_in_current_scope(ret[-1], var)
            if len(ret) >= 2:
                ret[-2].extend(ret[-1])
                ret.pop()  # The return space is deallocated.
            stack.pop()  # The stack frame is useless anymore,
            continue  #### and there is no need to procees the current stack frame any further

        stack[-1][-1] = True  # mark entered = True

        if recursion_depth == len(ret) + 1:
            # We recurse into a new stackframe, that call will have return value.
            # Set up the return space for it.
            ret.append([])
        else:
            fail("Invalid recursion_depth change, original depth", len(ret), "current depth", recursion_depth)

        if _can_be_expanded(fg, var):
            if fg.iterate_over != None:
                iterated_over_values = access(var, fg.iterate_over)
                if type(iterated_over_values) != "list":
                    fail(fg.iterate_over, "is not an iterable")

                path_list = fg.iterate_over.split(".")
                if len(fg.flags) != 0:  # expanding flags
                    # expanding flags should iterate in order, no more recursion involved
                    for value in iterated_over_values:
                        new_var = create_var_from_value(value, parent = var, path_list = path_list)
                        _eval_flags_or_flag_groups(stack, ret, fg, new_var, recursion_depth, parse_flag_cache)
                else:  # expanding flag_groups
                    # expanding flag_groups should iterate in reversed order due to recursion
                    for value in reversed(iterated_over_values):
                        new_var = create_var_from_value(value, parent = var, path_list = path_list)
                        _eval_flags_or_flag_groups(stack, ret, fg, new_var, recursion_depth, parse_flag_cache)
            else:
                _eval_flags_or_flag_groups(stack, ret, fg, var, recursion_depth, parse_flag_cache)

    if len(stack) != 0:
        fail("flag_group evaluation imcomplete")
    return ret

def eval_flag_group(fg, value, max_eval_iterations = 65536):
    ret = []
    _eval_flag_group_impl([], ret, fg, create_var_from_value(value), max_eval_iterations)
    processed_ret = []
    for flag_info in ret[0]:
        if len(flag_info.expandables) != 0:
            fail(flag_info, "is not fully expanded")
        processed_ret.append("".join(flag_info.chunks))
    return processed_ret

def eval_feature(feat, current_action, vars):
    ret = []
    if not feat.enabled:
        return ret
    enabled = False
    for fs in feat.flag_sets:
        if len(fs.with_features) != 0:
            fail("NotImplemented")
        if current_action in fs.actions:
            for fg in fs.flag_groups:
                pass
    return ret

feature = _feature
flag_group = _flag_group
flag_set = _flag_set
tool = _tool
tool_path = _tool_path
variable_with_value = _variable_with_value
