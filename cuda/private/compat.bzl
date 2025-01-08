_is_attr_string_keyed_label_dict_available = getattr(attr, "string_keyed_label_dict", None) != None
_is_bzlmod_enabled = str(Label("//:invalid")).startswith("@@")

def _attr(*args, **kwargs):
    """Compatibility layer for attr.string_keyed_label_dict(...)"""
    if _is_attr_string_keyed_label_dict_available:
        return attr.string_keyed_label_dict(*args, **kwargs)
    else:
        return attr.string_dict(*args, **kwargs)

def _repo_str(repo_str_or_repo_label):
    """Get mapped repo as string.

    Args:
        repo_str_or_repo_label: `"@repo"` or `Label("@repo")` """
    if type(repo_str_or_repo_label) == "Label":
        canonical_repo_name = repo_str_or_repo_label.repo_name
        repo_str = ("@@{}" if _is_bzlmod_enabled else "@{}").format(canonical_repo_name)
        return repo_str
    else:
        return repo_str_or_repo_label

components_mapping_compat = struct(
    attr = _attr,
    repo_str = _repo_str,
)
