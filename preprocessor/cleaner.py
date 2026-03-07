from __future__ import annotations
import re2
import orjson

_MULTIPART_NAME_RE = re2.compile(r'name="([^"]*)"')
_MULTIPART_FILENAME_RE = re2.compile(r'filename="')

_MAX_KEYS = 50
_MAX_DEPTH = 2


def extract_json_keys(body: bytes, content_type: str) -> list[str]:
    """Извлекает top-level ключи JSON-ответа для node features."""
    if "json" not in content_type.lower():
        return []
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return list(data.keys())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return list(data[0].keys())
    return []


def extract_json_values(body: bytes, content_type: str) -> dict[str, str]:
    """Извлекает leaf-значения из JSON для data dependency tracking."""
    if "json" not in content_type.lower():
        return {}
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError:
        return {}
    return _collect_leaf_values(data)


def _collect_leaf_values(obj: object, prefix: str = "", max_depth: int = _MAX_DEPTH, max_keys: int = _MAX_KEYS) -> dict[str, str]:
    """Рекурсивный обход объекта, возвращает {key: str(value)} для leaf-значений."""
    result: dict[str, str] = {}

    def _traverse(node: object, path: str, depth: int) -> None:
        if len(result) >= max_keys:
            return
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                child_path = f"{path}.{k}" if path else k
                _traverse(v, child_path, depth + 1)
        elif isinstance(node, list):
            if node:
                _traverse(node[0], path, depth + 1)
        else:
            if node is not None:
                result[path] = str(node)

    _traverse(obj, prefix, 0)
    return result


def extract_multipart_field_names(body: bytes, boundary: str) -> list[tuple[str, bool]]:
    """Extract field names from multipart/form-data body.

    Split by boundary, find Content-Disposition: form-data; name="..."
    is_file = True if filename= is present.
    Returns list of (name, is_file).
    """
    results: list[tuple[str, bool]] = []
    delimiter = f"--{boundary}".encode()
    parts = body.split(delimiter)
    # Skip preamble (first part) and epilogue (last part, usually "--\r\n")
    for part in parts[1:]:
        stripped = part.strip()
        if not stripped or stripped == b"--":
            continue
        # Decode headers section (before double CRLF)
        try:
            header_section = part.split(b"\r\n\r\n", 1)[0].decode("utf-8", errors="replace")
        except Exception:
            continue
        name_match = _MULTIPART_NAME_RE.search(header_section)
        if name_match is None:
            continue
        name = name_match.group(1)
        is_file = bool(_MULTIPART_FILENAME_RE.search(header_section))
        results.append((name, is_file))
    return results
