"""Static feature extraction for Smart Radar Graph Builder.

Extracts 32 features (indices 0-31) from a list of RawEndpoint objects
into an [N, 32] numpy array (STATIC_DIM = 32).

Features 32-34 (structural) are appended later by graph_builder.py.
"""
from __future__ import annotations

import re2 as re
import numpy as np

from smart_radar_graph.schema import RawEndpoint

STATIC_DIM = 32

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------
_ADMIN_SEGMENTS = frozenset({
    "admin", "manage", "management", "internal", "debug",
    "system", "console", "superuser",
})
_AUTH_SEGMENTS = frozenset({
    "auth", "login", "logout", "register", "signup", "signin",
    "token", "oauth", "sso", "password", "reset", "verify", "mfa", "2fa",
})
_USER_DATA_SEGMENTS = frozenset({
    "users", "accounts", "profile", "me", "user",
    "account", "customers", "members",
})
_PUBLIC_SEGMENTS = frozenset({
    "health", "status", "ping", "version", "docs",
    "swagger", "openapi", "metrics", "public",
})

# Prefixes stripped before computing resource_hierarchy_level
_VERSION_PREFIXES = frozenset({"api", "v1", "v2", "v3"})

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------
_URL_PARAM_RE = re.compile(
    r"(?i)(url|uri|link|callback|webhook|redirect|href|fetch|image_?url|next|return)"
)
_PAGINATION_RE = re.compile(
    r"(?i)^(page|limit|offset|cursor|per_?page|page_?size|next|previous|total|has_?more|count)$"
)
_SENSITIVE_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api_?key|ssn|credit_?card|card_?number|cvv|pin"
    r"|email|phone|dob|birth|salary|wage|role|is_?admin|permission|grant|scope"
    r"|jwt|session|private_?key)\b"
)
_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_literal_segments(path: str) -> list[str]:
    """Return path segments that are NOT path parameters (i.e. not {…})."""
    segments = [s for s in path.split("/") if s]
    return [s for s in segments if not _PATH_PARAM_RE.fullmatch(s)]


def _classify_content_type(ct: str | None) -> list[float]:
    """Return a one-hot vector for content type classification.

    For request: [json, form-urlencoded, multipart, xml-other]  (4d)
    For response: [json, html-xml, binary-other]  (3d) — caller selects slice.
    We return 4d here; callers take what they need.
    """
    if ct is None:
        return [0.0, 0.0, 0.0, 0.0]
    ct_lower = ct.lower()
    if "json" in ct_lower:
        return [1.0, 0.0, 0.0, 0.0]
    if "form-urlencoded" in ct_lower:
        return [0.0, 1.0, 0.0, 0.0]
    if "multipart" in ct_lower:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _classify_response_content_type(ct: str | None) -> list[float]:
    """3d one-hot: json, html-xml, binary-other."""
    if ct is None:
        return [0.0, 0.0, 0.0]
    ct_lower = ct.lower()
    if "json" in ct_lower:
        return [1.0, 0.0, 0.0]
    if "html" in ct_lower or "xml" in ct_lower:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _resource_hierarchy_level(path: str) -> float:
    """Compute resource hierarchy level (ordinal, capped at 3).

    Algorithm:
      1. Split path into segments, ignore empty strings.
      2. Strip leading api/version prefix segments (api, v1, v2, v3).
      3. Count remaining segments to determine level:
         - 0 remaining → 0 (only prefix)
         - 1 non-param segment, no path params → 1 (single resource)
         - first non-param segment + path param → 2
         - sub-resource or deeper → 3 (capped)
    """
    segments = [s for s in path.split("/") if s]

    # Strip api/version prefixes
    while segments and segments[0].lower() in _VERSION_PREFIXES:
        segments.pop(0)

    if not segments:
        return 0.0

    # Walk segments and classify
    level = 0
    for seg in segments:
        is_param = bool(_PATH_PARAM_RE.fullmatch(seg))
        if level == 0:
            if not is_param:
                level = 1  # first resource name
            # skip param-only leading segment (shouldn't happen normally)
        elif level == 1:
            if is_param:
                level = 2  # resource/{id}
            else:
                level = 3  # two resource names without param = sub-resource
                break
        elif level == 2:
            # anything after resource/{id} → sub-resource
            level = 3
            break

    return float(min(level, 3))


def _auth_type_onehot(auth_type: str) -> list[float]:
    """4d one-hot: none=0, basic=1, bearer-oauth=2, apikey=3."""
    mapping = {"none": 0, "basic": 1, "bearer-oauth": 2, "apikey": 3}
    idx = mapping.get(auth_type.lower(), 0)
    vec = [0.0, 0.0, 0.0, 0.0]
    vec[idx] = 1.0
    return vec


def _privilege_level(ep: RawEndpoint, literal_segs: set[str]) -> float:
    """Ordinal privilege level: 0=public, 1=user, 2=admin."""
    if literal_segs & _ADMIN_SEGMENTS:
        return 2.0
    if literal_segs & _PUBLIC_SEGMENTS and not ep.auth_required:
        return 0.0
    if ep.auth_required:
        return 1.0
    return 0.0


def _has_sensitive(ep: RawEndpoint) -> float:
    """Check all parameter and field names for sensitive data indicators."""
    names: list[str] = []
    names.extend(p.name for p in ep.query_params)
    names.extend(p.name for p in ep.header_params)
    names.extend(f.name for f in ep.request_body_fields)
    names.extend(f.name for f in ep.response_body_fields)
    names.extend(ep.path_params)
    for name in names:
        if _SENSITIVE_RE.search(name):
            return 1.0
    return 0.0


def _extract_single(ep: RawEndpoint) -> list[float]:
    """Extract all 32 static features for one endpoint."""
    feats: list[float] = []

    # ── 0-4: HTTP method one-hot (GET, POST, PUT/PATCH, DELETE, OTHER) ──
    method = ep.method.upper()
    method_vec = [0.0] * 5
    if method == "GET":
        method_vec[0] = 1.0
    elif method == "POST":
        method_vec[1] = 1.0
    elif method in ("PUT", "PATCH"):
        method_vec[2] = 1.0
    elif method == "DELETE":
        method_vec[3] = 1.0
    else:
        method_vec[4] = 1.0
    feats.extend(method_vec)  # indices 0-4

    # ── 5: path_depth (number of non-empty segments) ──
    segments_all = [s for s in ep.path_template.split("/") if s]
    feats.append(float(len(segments_all)))  # index 5

    # ── 6: num_path_params (count of {…} segments) ──
    num_path_params = len(_PATH_PARAM_RE.findall(ep.path_template))
    feats.append(float(num_path_params))  # index 6

    # ── 7-9: resource_category multi-hot (admin, auth, user-data) ──
    literal_segs = set(_get_literal_segments(ep.path_template))
    feats.append(1.0 if literal_segs & _ADMIN_SEGMENTS else 0.0)     # 7
    feats.append(1.0 if literal_segs & _AUTH_SEGMENTS else 0.0)      # 8
    feats.append(1.0 if literal_segs & _USER_DATA_SEGMENTS else 0.0) # 9

    # ── 10: num_query_params ──
    feats.append(float(len(ep.query_params)))  # 10

    # ── 11: num_header_params ──
    feats.append(float(len(ep.header_params)))  # 11

    # ── 12: has_request_body ──
    has_body = (ep.request_content_type is not None) or (len(ep.request_body_fields) > 0)
    feats.append(1.0 if has_body else 0.0)  # 12

    # ── 13-16: request_content_type one-hot (json, form-urlencoded, multipart, xml-other) ──
    feats.extend(_classify_content_type(ep.request_content_type))  # 13-16

    # ── 17: input_complexity = (path_params + query + header + body_top_level) * (1 + max_nesting) ──
    body_top_level = sum(1 for f in ep.request_body_fields if f.depth == 0)
    total_inputs = num_path_params + len(ep.query_params) + len(ep.header_params) + body_top_level
    max_nesting = max((f.depth for f in ep.request_body_fields), default=0)
    input_complexity = float(total_inputs * max(1, max_nesting))
    feats.append(input_complexity)  # 17

    # ── 18: has_url_param ──
    has_url = any(_URL_PARAM_RE.search(p.name) for p in ep.query_params)
    feats.append(1.0 if has_url else 0.0)  # 18

    # ── 19-22: auth_type one-hot (none, basic, bearer-oauth, apikey) ──
    feats.extend(_auth_type_onehot(ep.auth_type))  # 19-22

    # ── 23: auth_required ──
    feats.append(1.0 if ep.auth_required else 0.0)  # 23

    # ── 24: privilege_level (ordinal 0/1/2) ──
    feats.append(_privilege_level(ep, literal_segs))  # 24

    # ── 25-27: response_content_type one-hot (json, html-xml, binary-other) ──
    feats.extend(_classify_response_content_type(ep.response_content_type))  # 25-27

    # ── 28: response_is_collection ──
    feats.append(1.0 if ep.response_is_array else 0.0)  # 28

    # ── 29: has_pagination ──
    has_pag = any(_PAGINATION_RE.match(p.name) for p in ep.query_params)
    feats.append(1.0 if has_pag else 0.0)  # 29

    # ── 30: has_sensitive_indicator ──
    feats.append(_has_sensitive(ep))  # 30

    # ── 31: resource_hierarchy_level (ordinal 0..3+) ──
    feats.append(_resource_hierarchy_level(ep.path_template))  # 31

    assert len(feats) == STATIC_DIM, f"Expected {STATIC_DIM} features, got {len(feats)}"
    return feats


def extract_static_features(endpoints: list[RawEndpoint]) -> np.ndarray:
    """Extract static features for a list of endpoints.

    Args:
        endpoints: list of RawEndpoint objects (N items)

    Returns:
        np.ndarray of shape [N, STATIC_DIM] with dtype float32
    """
    rows = [_extract_single(ep) for ep in endpoints]
    return np.array(rows, dtype=np.float32)
