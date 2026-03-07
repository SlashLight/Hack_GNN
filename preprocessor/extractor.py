from __future__ import annotations
import html
import math
import re2
import orjson
from collections import Counter
from urllib.parse import parse_qs, unquote

from preprocessor.models import Parameter

_BOOLEAN_OPTIONS = re2.Options()
_BOOLEAN_OPTIONS.case_sensitive = False

_TYPE_PATTERNS: list[tuple[str, re2.Pattern]] = [
    ("jwt",      re2.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$")),
    ("email",    re2.compile(r"^[^@]+@[^@]+\.[a-zA-Z]{2,}$")),
    ("uuid",     re2.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")),
    ("objectid", re2.compile(r"^[0-9a-f]{24}$")),
    ("url",      re2.compile(r"^https?://")),
    ("date",     re2.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("boolean",  re2.compile(r"^(true|false|0|1)$", options=_BOOLEAN_OPTIONS)),
    ("int",      re2.compile(r"^\d+$")),
    ("hash",     re2.compile(r"^[0-9a-f]{32,}$")),
    ("base64",   re2.compile(r"^(?:[A-Za-z0-9+/\-_]{4}){8,}={0,2}$")),
]

_MAX_VALUE_LEN = 256
_MAX_DICT_KEYS = 50
_MAX_PARAMS = 50

_MIN_REFLECTION_LEN = 4  # значения длиннее 3 символов

_SENSITIVE_PATTERNS = [
    "password", "passwd", "pwd", "secret", "token", "api_key",
    "apikey", "auth", "session", "csrf", "ssn", "credit_card",
    "private", "key", "access_token", "refresh_token",
]

_DYNAMIC_SEGMENT_RE = re2.compile(r"^\{[A-Z]+\}$")


def _is_reflected(value: str, response_body: str) -> bool:
    """Проверяет reflection с учётом encoding-вариантов."""
    if len(value) < _MIN_REFLECTION_LEN:
        return False

    # 1. Exact match
    if value in response_body:
        return True

    # 2. Case-insensitive
    if value.lower() in response_body.lower():
        return True

    # 3. HTML entity decoded body
    body_unescaped = html.unescape(response_body)
    if value.lower() in body_unescaped.lower():
        return True

    # 4. URL-decoded body
    body_url_decoded = unquote(response_body)
    if value.lower() in body_url_decoded.lower():
        return True

    return False


def infer_type(value: str) -> str:
    """Определяет тип значения параметра."""
    for type_name, pattern in _TYPE_PATTERNS:
        if pattern.match(value):
            return type_name
    return "string"


def compute_entropy(value: str) -> float:
    """Энтропия Шеннона строки. Высокая → случайные данные (токены, хэши)."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def is_sensitive_name(name: str) -> bool:
    """Проверяет, является ли имя параметра чувствительным."""
    lower = name.lower()
    return any(p in lower for p in _SENSITIVE_PATTERNS)


def extract_auth(headers: dict[str, str]) -> str | None:
    lower = {k.lower(): v for k, v in headers.items()}
    auth = lower.get("authorization", "")
    if auth.lower().startswith("bearer"):
        return "Bearer"
    if "cookie" in lower:
        return "Cookie"
    return None


def extract_path_params(original_path: str, normalized_path: str) -> list[Parameter]:
    """Extract path parameters by comparing original and normalized paths.

    Split both by "/", compare segments.
    Where normalized has {INT}/{ID}/{HASH}/{DATE} → create Parameter
    name=f"path_{index}", location="path"
    """
    params: list[Parameter] = []
    orig_segments = original_path.split("/")
    norm_segments = normalized_path.split("/")
    if len(orig_segments) != len(norm_segments):
        return params
    for index, (orig, norm) in enumerate(zip(orig_segments, norm_segments)):
        if _DYNAMIC_SEGMENT_RE.match(norm):
            # Derive the type from the placeholder name
            placeholder = norm[1:-1].lower()  # strip braces, lowercase
            if placeholder == "int":
                value_type = "int"
            elif placeholder == "uuid" or placeholder == "id":
                # Try to infer from actual value
                value_type = infer_type(orig)
            elif placeholder == "hash":
                value_type = "hash"
            elif placeholder == "date":
                value_type = "date"
            else:
                value_type = infer_type(orig)
            params.append(Parameter(
                name=f"path_{index}",
                value_type=value_type,
                location="path",
                is_reflected=False,
                value_entropy=compute_entropy(orig),
                is_sensitive=False,
                original_value=orig,
            ))
    return params


def extract_parameters(
    query_string: str,
    json_body: bytes | None,
    form_data: dict[str, list[str]] | None,
    response_body: str,
    multipart_fields: list[tuple[str, bool]] | None = None,
) -> list[Parameter]:
    raw: list[tuple[str, str, str, bool]] = []  # (name, value, location, truncated)
    seen: set[str] = set()  # дедупликация по имени

    # Query string → location="query"
    for name, values in parse_qs(query_string, keep_blank_values=True).items():
        if name not in seen:
            seen.add(name)
            val = values[0]
            trunc = len(val) > _MAX_VALUE_LEN
            if trunc:
                val = val[:_MAX_VALUE_LEN]
            raw.append((name, val, "query", trunc))

    # JSON body → location="body"
    if json_body:
        try:
            data = orjson.loads(json_body)
            _collect_json_params(data, raw, seen, depth=0)
        except orjson.JSONDecodeError:
            pass

    # Form data → location="body"
    if form_data:
        for name, values in form_data.items():
            if name not in seen:
                seen.add(name)
                val = values[0] if values else ""
                trunc = len(val) > _MAX_VALUE_LEN
                if trunc:
                    val = val[:_MAX_VALUE_LEN]
                raw.append((name, val, "body", trunc))

    params = []
    for name, value, location, truncated in raw:
        reflected = False if truncated else _is_reflected(value, response_body)
        params.append(Parameter(
            name=name,
            value_type=infer_type(value),
            location=location,
            is_reflected=reflected,
            value_entropy=compute_entropy(value),
            is_sensitive=is_sensitive_name(name),
        ))

    # Multipart field names → location="body" (value_type determined by is_file flag)
    if multipart_fields:
        for name, is_file in multipart_fields:
            if name not in seen:
                seen.add(name)
                params.append(Parameter(
                    name=name,
                    value_type="file" if is_file else "string",
                    location="body",
                    is_reflected=False,
                    value_entropy=0.0,
                    is_sensitive=is_sensitive_name(name),
                ))

    return params[:_MAX_PARAMS]


def _collect_json_params(
    obj: object,
    out: list[tuple[str, str, str, bool]],
    seen: set[str],
    depth: int,
) -> None:
    if depth > 2 or len(seen) >= _MAX_PARAMS:
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:_MAX_DICT_KEYS]:
            if isinstance(v, (str, int, float, bool)):
                if k not in seen:
                    seen.add(k)
                    val = str(v)
                    trunc = len(val) > _MAX_VALUE_LEN
                    if trunc:
                        val = val[:_MAX_VALUE_LEN]
                    out.append((k, val, "body", trunc))
            elif isinstance(v, (dict, list)):
                _collect_json_params(v, out, seen, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:3]:
            _collect_json_params(item, out, seen, depth + 1)


