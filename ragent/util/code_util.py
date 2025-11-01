import re


def get_code_text_from_path(node, path_parts):
    if len(path_parts) == 1:
        return "\n".join(node[path_parts[0]]["text"])

    return get_code_text_from_path(node[path_parts[0]], path_parts[1:])


def extract_patch_file_path(patch_str):
    """Extract the file path that was patched from the diff."""
    match = re.search(r"^diff --git a/(.*?) b/\1", patch_str, re.MULTILINE)
    if match:
        return match.group(1)
    fallback = re.search(r"^\+\+\+ b/(.+)", patch_str, re.MULTILINE)
    if fallback:
        return fallback.group(1)
    return None
