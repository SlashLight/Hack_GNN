"""OpenAPI 3.0/3.1 extractor — Mode A.

Parses an OpenAPI spec (dict or file path/str) into a list of RawEndpoint
objects for downstream graph building and feature extraction.
"""
from __future__ import annotations

import re2 as re
from pathlib import Path
from typing import Any

from smart_radar_graph.schema import FieldInfo, ParamInfo, RawEndpoint

# ---------------------------------------------------------------------------
# Standard HTTP headers to exclude from header_params
# ---------------------------------------------------------------------------
_STANDARD_HEADERS: frozenset[str] = frozenset({
    "accept",
    "accept-charset",
    "accept-encoding",
    "accept-language",
    "authorization",
    "cache-control",
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "cookie",
    "date",
    "expect",
    "forwarded",
    "from",
    "host",
    "if-match",
    "if-modified-since",
    "if-none-match",
    "if-range",
    "if-unmodified-since",
    "max-forwards",
    "origin",
    "pragma",
    "proxy-authorization",
    "range",
    "referer",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "user-agent",
    "via",
    "warning",
})

# HTTP methods that OpenAPI uses for operations
_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})

# 2xx response codes to look for (in preference order)
_SUCCESS_CODES = ("200", "201", "202", "204")

# Max depth for JSON Schema recursion
_MAX_BODY_DEPTH = 3


# ---------------------------------------------------------------------------
# Helper: extract path params from template string
# ---------------------------------------------------------------------------

def _extract_path_params(path_template: str) -> list[str]:
    """Return parameter names WITHOUT braces: '/users/{id}' → ['id']."""
    return re.findall(r'\{([^}]+)\}', path_template)


# ---------------------------------------------------------------------------
# Helper: JSON Schema type extraction
# ---------------------------------------------------------------------------

def _schema_type(schema: dict) -> str:
    """Return a normalised type string from a JSON Schema dict."""
    return schema.get("type", "string")


# ---------------------------------------------------------------------------
# Helper: recurse into JSON Schema to collect FieldInfo list
# ---------------------------------------------------------------------------

def _extract_fields(schema: dict, depth: int, max_depth: int) -> list[FieldInfo]:
    """Recursively extract named fields from a JSON Schema up to *max_depth*.

    - ``type: object`` → iterate over ``properties``
    - ``type: array``  → recurse into ``items`` (without incrementing depth for
      the array itself; the items' properties get the current depth)
    """
    if schema is None or depth > max_depth:
        return []

    schema_type = _schema_type(schema)
    fields: list[FieldInfo] = []

    if schema_type == "object":
        properties: dict = schema.get("properties", {})
        for name, prop_schema in properties.items():
            prop_type = _schema_type(prop_schema)
            fields.append(FieldInfo(name=name, type=prop_type, depth=depth))
            # Recurse into nested objects/arrays
            fields.extend(_extract_fields(prop_schema, depth + 1, max_depth))

    elif schema_type == "array":
        items_schema = schema.get("items", {})
        # Recurse into array items at the *same* depth (the array wrapper isn't
        # a named field itself when we're already inside a property context)
        fields.extend(_extract_fields(items_schema, depth, max_depth))

    return fields


# ---------------------------------------------------------------------------
# Helper: resolve auth info
# ---------------------------------------------------------------------------

def _resolve_auth(
    operation: dict,
    global_security: list[dict],
    security_schemes: dict,
) -> tuple[str, bool]:
    """Return (auth_type, auth_required) for one operation.

    Operation-level ``security`` overrides global.  An empty list means no
    auth required.
    """
    # Determine which security requirements apply
    if "security" in operation:
        effective_security = operation["security"]
    else:
        effective_security = global_security

    if not effective_security:
        return "none", False

    # Resolve the first security requirement to a type
    first_scheme_name = next(iter(effective_security[0]), None)
    if first_scheme_name is None:
        return "none", False

    scheme_def = security_schemes.get(first_scheme_name, {})
    scheme_type = scheme_def.get("type", "").lower()
    scheme_scheme = scheme_def.get("scheme", "").lower()

    if scheme_type == "oauth2":
        return "bearer-oauth", True
    if scheme_type == "http" and scheme_scheme == "bearer":
        return "bearer-oauth", True
    if scheme_type == "http" and scheme_scheme == "basic":
        return "basic", True
    if scheme_type == "apikey":
        return "apikey", True

    # Unknown / unrecognised → treat as none
    return "none", False


# ---------------------------------------------------------------------------
# Helper: extract parameters (query / header / path) from merged list
# ---------------------------------------------------------------------------

def _extract_parameters(
    merged_params: list[dict],
) -> tuple[list[str], list[ParamInfo], list[ParamInfo]]:
    """Return (extra_path_params_from_spec, query_params, header_params).

    Path params from ``in: path`` are returned as plain names (no braces).
    Standard headers are filtered out.
    """
    query_params: list[ParamInfo] = []
    header_params: list[ParamInfo] = []
    path_param_names: list[str] = []

    for param in merged_params:
        name: str = param.get("name", "")
        location: str = param.get("in", "")
        schema: dict = param.get("schema", {})
        param_type = _schema_type(schema) if schema else "string"

        if location == "query":
            query_params.append(ParamInfo(name=name, type=param_type))
        elif location == "header":
            if name.lower() not in _STANDARD_HEADERS:
                header_params.append(ParamInfo(name=name, type=param_type))
        elif location == "path":
            path_param_names.append(name)

    return path_param_names, query_params, header_params


# ---------------------------------------------------------------------------
# Helper: extract request body fields + content type
# ---------------------------------------------------------------------------

def _extract_request_body(
    operation: dict,
) -> tuple[str | None, list[FieldInfo]]:
    """Return (content_type, fields) from the requestBody, if present."""
    request_body: dict | None = operation.get("requestBody")
    if not request_body:
        return None, []

    content: dict = request_body.get("content", {})
    if not content:
        return None, []

    # Prefer application/json; fall back to first available
    content_type: str
    if "application/json" in content:
        content_type = "application/json"
    else:
        content_type = next(iter(content))

    schema: dict = content[content_type].get("schema", {})
    fields = _extract_fields(schema, depth=0, max_depth=_MAX_BODY_DEPTH)
    return content_type, fields


# ---------------------------------------------------------------------------
# Helper: find first 2xx response and extract its schema
# ---------------------------------------------------------------------------

def _extract_response(
    operation: dict,
) -> tuple[str | None, list[FieldInfo], bool]:
    """Return (content_type, fields, response_is_array) for first 2xx response."""
    responses: dict = operation.get("responses", {})

    for code in _SUCCESS_CODES:
        if code not in responses:
            continue
        response_obj: dict = responses[code]
        content: dict = response_obj.get("content", {})
        if not content:
            return None, [], False

        if "application/json" in content:
            content_type = "application/json"
        else:
            content_type = next(iter(content))

        schema: dict = content[content_type].get("schema", {})
        if not schema:
            return content_type, [], False

        schema_type = _schema_type(schema)
        response_is_array = schema_type == "array"

        fields = _extract_fields(schema, depth=0, max_depth=_MAX_BODY_DEPTH)
        return content_type, fields, response_is_array

    return None, [], False


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_from_spec(spec: dict | str | Path) -> list[RawEndpoint]:
    """Parse an OpenAPI 3.0/3.1 spec into a list of RawEndpoint objects.

    Parameters
    ----------
    spec:
        Either a pre-parsed dict (useful for tests) or a file path / URL
        string that will be loaded with ``prance.ResolvingParser`` for
        full ``$ref`` resolution.
    """
    if isinstance(spec, (str, Path)):
        try:
            import prance  # lazy import — only needed for file loading
        except ImportError as exc:
            raise ImportError(
                "prance is required to load OpenAPI specs from files. "
                "Install it with: pip install prance openapi-spec-validator"
            ) from exc
        parser = prance.ResolvingParser(str(spec), lazy=False, strict=False)
        spec_dict: dict = parser.specification
    else:
        spec_dict = spec

    # ── Global security + schemes ────────────────────────────────────────────
    global_security: list[dict] = spec_dict.get("security", [])
    components: dict = spec_dict.get("components", {})
    security_schemes: dict = components.get("securitySchemes", {})

    endpoints: list[RawEndpoint] = []

    paths: dict = spec_dict.get("paths", {})
    for path_template, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Path-level parameters (shared across all operations on this path)
        path_level_params: list[dict] = path_item.get("parameters", [])

        # Path params extracted directly from the URL template
        template_path_params: list[str] = _extract_path_params(path_template)

        for method_lower, operation in path_item.items():
            if method_lower not in _HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                continue

            method = method_lower.upper()

            # ── Merge parameters: path-level overridden by operation-level ──
            operation_params: list[dict] = operation.get("parameters", [])
            # Build a dict keyed by (name, in); operation overrides path-level
            merged: dict[tuple[str, str], dict] = {}
            for p in path_level_params:
                key = (p.get("name", ""), p.get("in", ""))
                merged[key] = p
            for p in operation_params:
                key = (p.get("name", ""), p.get("in", ""))
                merged[key] = p

            _spec_path_params, query_params, header_params = _extract_parameters(
                list(merged.values())
            )

            # Final path params: from template (canonical) + any extra from spec
            # that weren't already in the template (edge case)
            all_path_params: list[str] = list(template_path_params)
            for name in _spec_path_params:
                if name not in all_path_params:
                    all_path_params.append(name)

            # ── Request body ─────────────────────────────────────────────────
            req_content_type, req_fields = _extract_request_body(operation)

            # ── Response ─────────────────────────────────────────────────────
            resp_content_type, resp_fields, resp_is_array = _extract_response(operation)

            # ── Auth ─────────────────────────────────────────────────────────
            auth_type, auth_required = _resolve_auth(
                operation, global_security, security_schemes
            )

            endpoints.append(
                RawEndpoint(
                    path_template=path_template,
                    method=method,
                    path_params=all_path_params,
                    query_params=query_params,
                    header_params=header_params,
                    request_content_type=req_content_type,
                    request_body_fields=req_fields,
                    response_content_type=resp_content_type,
                    response_body_fields=resp_fields,
                    response_is_array=resp_is_array,
                    auth_type=auth_type,
                    auth_required=auth_required,
                )
            )

    return endpoints
