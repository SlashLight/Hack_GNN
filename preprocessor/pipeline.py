from __future__ import annotations
import hashlib
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

from preprocessor.normalizer import normalize_url

_SESSION_COOKIE_NAMES = frozenset({
    "session", "sid", "jsessionid", "phpsessid",
    "connect.sid", "_session", "auth_token", "sess",
})
from preprocessor.cleaner import extract_multipart_field_names, extract_json_keys, extract_json_values
from preprocessor.extractor import (
    extract_parameters,
    extract_auth,
    extract_path_params,
)
from preprocessor.models import Parameter, ProcessedEvent


@dataclass
class RawHTTPEvent:
    """Сырое HTTP-событие из mitmproxy или любого другого источника."""
    method: str
    url: str
    headers: dict[str, str]
    request_body: bytes | None
    status_code: int
    response_headers: dict[str, str]
    response_body: bytes
    timestamp: float = 0.0


def _extract_session_id(headers: dict[str, str]) -> str:
    """Возвращает md5[:12] хеш токена или session cookie. Fallback — 'anonymous'."""
    # 1. Bearer token — primary
    auth = headers.get("authorization", "") or headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):]
        if token:
            return hashlib.md5(token.encode()).hexdigest()[:12]

    # 2. Session cookie — fallback
    cookie = headers.get("cookie", "") or headers.get("Cookie", "")
    for part in cookie.split(";"):
        part = part.strip()
        if "=" in part:
            name, value = part.split("=", 1)
            if name.strip().lower() in _SESSION_COOKIE_NAMES:
                return hashlib.md5(value.strip().encode()).hexdigest()[:12]

    return "anonymous"


def _make_processed_event(raw: RawHTTPEvent) -> ProcessedEvent:
    """Конвертирует RawHTTPEvent → ProcessedEvent через весь пайплайн."""
    # 1. Normalize URL
    parsed_url = urlparse(raw.url)
    original_path = parsed_url.path
    path = normalize_url(parsed_url.path)
    endpoint_id = f"{raw.method} {path}"

    # 2. Extract parameters
    query = parsed_url.query
    json_body: bytes | None = None
    form_data: dict[str, list[str]] | None = None
    req_ct = raw.headers.get("content-type", "") or raw.headers.get("Content-Type", "")

    multipart_fields = None
    if "multipart/form-data" in req_ct:
        boundary = ""
        for part in req_ct.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary="):].strip('"')
                break
        if boundary and raw.request_body:
            multipart_fields = extract_multipart_field_names(raw.request_body, boundary)
    elif "json" in req_ct:
        json_body = raw.request_body
    elif "form" in req_ct:
        if raw.request_body:
            form_str = raw.request_body.decode("utf-8", errors="replace")
            form_data = parse_qs(form_str, keep_blank_values=True)

    # Reflection check requires response body text
    response_text = raw.response_body.decode("utf-8", errors="replace")
    params = extract_parameters(query, json_body, form_data, response_text, multipart_fields=multipart_fields)

    # Add path params
    path_params = extract_path_params(original_path, path)
    params = params + path_params

    # 3. Auth
    auth = extract_auth(raw.headers)

    # 4. Response content-type
    ct = raw.response_headers.get("content-type", "") or raw.response_headers.get("Content-Type", "")

    # 5. JSON keys & values from response body
    resp_keys = extract_json_keys(raw.response_body, ct)
    resp_values = extract_json_values(raw.response_body, ct)

    # 6. Session id
    session_id = _extract_session_id(raw.headers)

    return ProcessedEvent(
        endpoint_id=endpoint_id,
        method=raw.method,
        has_dynamic_segment=any(p in path for p in ("{INT}", "{ID}", "{HASH}", "{DATE}")),
        auth_type=auth,
        parameters=params,
        timestamp=raw.timestamp,
        original_path=original_path,
        session_id=session_id,
        response_keys=resp_keys,
        response_values=resp_values,
        status_code=raw.status_code,
        content_type=ct,
    )
