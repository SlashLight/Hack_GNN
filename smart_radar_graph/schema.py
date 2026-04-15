from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ParamInfo:
    """Parameter name + type. Used for query params and header params."""
    name: str
    type: str = "string"  # "string"|"integer"|"number"|"boolean"|"array"|"object"|"file"


@dataclass
class FieldInfo:
    """A field in request/response body with nesting depth."""
    name: str
    type: str
    depth: int  # 0 = top-level, 1 = nested one level, ...


@dataclass
class RawEndpoint:
    """Normalized endpoint — shared contract between extractors and graph backend.

    Both Mode A (OpenAPI) and Mode B (traffic) produce these objects.
    The graph builder never knows the source.
    """
    path_template: str                          # "/api/v1/users/{id}/posts"
    method: str                                 # "GET" | "POST" | ...
    path_params: list[str] = field(default_factory=list)            # ["id"] — names WITHOUT braces
    query_params: list[ParamInfo] = field(default_factory=list)
    header_params: list[ParamInfo] = field(default_factory=list)    # custom only
    request_content_type: str | None = None
    request_body_fields: list[FieldInfo] = field(default_factory=list)
    response_content_type: str | None = None
    response_body_fields: list[FieldInfo] = field(default_factory=list)
    response_is_array: bool = False
    auth_type: str = "none"                     # "none"|"basic"|"bearer-oauth"|"apikey"
    auth_required: bool = False
    raw_examples: list[dict] = field(default_factory=list)
