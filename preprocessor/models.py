from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Parameter:
    """Параметр HTTP-запроса, извлечённый из query string, JSON body или form-data."""
    name: str
    value_type: str           # "int" | "uuid" | "string" | "jwt" | "email" | "hash"
    location: str = "query"   # "query" | "path" | "body" | "header" | "cookie"
    is_reflected: bool = False
    value_entropy: float = 0.0    # Shannon entropy of the value
    is_sensitive: bool = False    # name contains password/token/secret/key
    original_value: str | None = None  # Оригинальное значение параметра (path/query/body)


@dataclass
class ProcessedEvent:
    """Нормализованное HTTP-событие, готовое для добавления в граф."""
    endpoint_id: str              # "POST /api/users/{INT}/upload"
    method: str = "GET"           # "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
    has_dynamic_segment: bool = False  # contains {INT}, {ID}, {HASH}, {DATE}
    auth_type: str | None = None  # "Bearer" | "Cookie" | "Basic" | "ApiKey" | None
    parameters: list[Parameter] = field(default_factory=list)
    timestamp: float = 0.0        # unix timestamp of the request
    original_path: str = ""       # URL before normalization
    session_id: str = ""          # md5[:12] of Bearer/cookie, "anonymous" as fallback
    response_keys: list[str] = field(default_factory=list)    # top-level JSON keys
    response_values: dict[str, str] = field(default_factory=dict)  # leaf values for data-dep tracking
    status_code: int = 0
    content_type: str = ""
