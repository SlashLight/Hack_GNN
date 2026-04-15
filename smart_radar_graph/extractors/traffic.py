"""Mode B extractor: converts ProcessedEvent traffic into RawEndpoint objects."""
from __future__ import annotations

import re
from collections import Counter

from preprocessor.models import ProcessedEvent
from smart_radar_graph.schema import FieldInfo, ParamInfo, RawEndpoint

# Auth type mapping from ProcessedEvent.auth_type → RawEndpoint.auth_type
_AUTH_MAP: dict[str | None, str] = {
    "Bearer": "bearer-oauth",
    "Cookie": "apikey",
    "Basic": "basic",
    None: "none",
}

_AUTH_REQUIRED_THRESHOLD = 0.8


def _parse_method_path(endpoint_id: str) -> tuple[str, str]:
    """Split 'METHOD /path/template' into (method, path)."""
    parts = endpoint_id.split(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "GET", endpoint_id


def _most_common(values: list) -> object:
    """Return the most common element in a list, or None if empty."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def extract_from_traffic(
    endpoint_events: dict[str, list[ProcessedEvent]],
) -> list[RawEndpoint]:
    """Convert grouped ProcessedEvent traffic into a list of RawEndpoint objects.

    Args:
        endpoint_events: Mapping of endpoint_id (e.g. "GET /api/users/{INT}")
                         to the list of ProcessedEvent objects observed for it.

    Returns:
        One RawEndpoint per key in endpoint_events.
    """
    result: list[RawEndpoint] = []

    for endpoint_id, events in endpoint_events.items():
        method, path = _parse_method_path(endpoint_id)

        # --- path_params: strip braces from template placeholders ---
        path_params: list[str] = re.findall(r'\{([^}]+)\}', path)

        # --- query / header / body params: union across events, dedup by name ---
        query_seen: dict[str, ParamInfo] = {}
        header_seen: dict[str, ParamInfo] = {}
        body_fields_seen: dict[str, FieldInfo] = {}

        for ev in events:
            for param in ev.parameters:
                if param.location == "query":
                    if param.name not in query_seen:
                        query_seen[param.name] = ParamInfo(
                            name=param.name,
                            type=param.value_type,
                        )
                elif param.location == "header":
                    if param.name not in header_seen:
                        header_seen[param.name] = ParamInfo(
                            name=param.name,
                            type=param.value_type,
                        )
                elif param.location == "body":
                    if param.name not in body_fields_seen:
                        body_fields_seen[param.name] = FieldInfo(
                            name=param.name,
                            type=param.value_type,
                            depth=0,
                        )

        # --- content_type: most common non-empty ---
        # ProcessedEvent.content_type stores the response Content-Type
        content_types = [ev.content_type for ev in events if ev.content_type]
        content_type: str | None = _most_common(content_types) if content_types else None  # type: ignore[assignment]
        request_content_type = content_type
        response_content_type = content_type

        # --- response_body_fields: union of response_keys across events ---
        resp_fields_seen: dict[str, FieldInfo] = {}
        for ev in events:
            for key in ev.response_keys:
                if key not in resp_fields_seen:
                    resp_fields_seen[key] = FieldInfo(name=key, type="string", depth=0)

        # --- auth_type: map + most common in group ---
        mapped_auth = [_AUTH_MAP.get(ev.auth_type, "none") for ev in events]
        auth_type: str = _most_common(mapped_auth) or "none"  # type: ignore[assignment]

        # --- auth_required: ≥80% of events had non-None auth_type ---
        auth_count = sum(1 for ev in events if ev.auth_type is not None)
        auth_required = (auth_count / len(events)) >= _AUTH_REQUIRED_THRESHOLD if events else False

        # --- raw_examples: up to 5 ---
        raw_examples: list[dict] = []
        for ev in events[:5]:
            example: dict = {}
            if ev.parameters:
                example["params"] = {p.name: p.original_value for p in ev.parameters}
            if ev.response_values:
                example["response_values"] = ev.response_values
            raw_examples.append(example)

        result.append(RawEndpoint(
            path_template=path,
            method=method,
            path_params=path_params,
            query_params=list(query_seen.values()),
            header_params=list(header_seen.values()),
            request_content_type=request_content_type,
            request_body_fields=list(body_fields_seen.values()),
            response_content_type=response_content_type,
            response_body_fields=list(resp_fields_seen.values()),
            response_is_array=False,
            auth_type=auth_type,
            auth_required=auth_required,
            raw_examples=raw_examples,
        ))

    return result
