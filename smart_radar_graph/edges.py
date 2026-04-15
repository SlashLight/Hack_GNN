"""Edge builders for the Smart Radar heterogeneous graph.

Each function returns a torch.Tensor of shape [2, E] (COO format).
For undirected edges, BOTH directions (A→B and B→A) are stored.
Returns [2, 0] if no edges are found.
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

import torch

from smart_radar_graph.schema import RawEndpoint

# Type alias: (path_template, method) → node index
NodeIndex = dict[tuple[str, str], int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty() -> torch.Tensor:
    """Return an empty [2, 0] edge tensor."""
    return torch.zeros((2, 0), dtype=torch.long)


def _segments(path: str) -> list[str]:
    """Split a path template into non-empty segments."""
    return [s for s in path.split("/") if s]


def _is_param(seg: str) -> bool:
    """Return True if the segment is a path parameter like {id}."""
    return seg.startswith("{") and seg.endswith("}")


# ---------------------------------------------------------------------------
# Edge 1: resource_hierarchy (parent_of) — directed
# ---------------------------------------------------------------------------

def build_resource_hierarchy_edges(
    endpoints: list[RawEndpoint], idx: NodeIndex
) -> torch.Tensor:
    """Trie-based parent_of edges.

    For each (path_template, child), find the nearest prefix parent path.
    Operates at path_template level; if parent has multiple methods, create
    one edge per parent-method → child-method combination.
    """
    # Group node indices by path_template
    path_to_nodes: dict[str, list[int]] = defaultdict(list)
    for ep in endpoints:
        node_idx = idx.get((ep.path_template, ep.method))
        if node_idx is not None:
            path_to_nodes[ep.path_template].append(node_idx)

    unique_paths = sorted(path_to_nodes.keys())

    src_list: list[int] = []
    dst_list: list[int] = []

    segs_map = {p: _segments(p) for p in unique_paths}

    for child_path in unique_paths:
        child_segs = segs_map[child_path]
        best_parent: str | None = None
        best_len = -1

        for parent_path in unique_paths:
            if parent_path == child_path:
                continue
            parent_segs = segs_map[parent_path]
            # Parent must be strictly shorter
            if len(parent_segs) >= len(child_segs):
                continue
            # Child must start with parent segments
            if child_segs[: len(parent_segs)] == parent_segs:
                if len(parent_segs) > best_len:
                    best_len = len(parent_segs)
                    best_parent = parent_path

        if best_parent is not None:
            # Cartesian product of parent methods × child methods
            for p_node in path_to_nodes[best_parent]:
                for c_node in path_to_nodes[child_path]:
                    src_list.append(p_node)
                    dst_list.append(c_node)

    if not src_list:
        return _empty()
    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ---------------------------------------------------------------------------
# Edge 2: crud_with — undirected (both directions stored)
# ---------------------------------------------------------------------------

def build_crud_edges(
    endpoints: list[RawEndpoint], idx: NodeIndex
) -> torch.Tensor:
    """Same path_template, different HTTP methods → full clique (both directions)."""
    path_to_nodes: dict[str, list[int]] = defaultdict(list)
    for ep in endpoints:
        node_idx = idx.get((ep.path_template, ep.method))
        if node_idx is not None:
            path_to_nodes[ep.path_template].append(node_idx)

    src_list: list[int] = []
    dst_list: list[int] = []

    for nodes in path_to_nodes.values():
        if len(nodes) < 2:
            continue
        for a, b in combinations(nodes, 2):
            src_list.extend([a, b])
            dst_list.extend([b, a])

    if not src_list:
        return _empty()
    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ---------------------------------------------------------------------------
# Edge 3: sibling_of — undirected (both directions stored)
# ---------------------------------------------------------------------------

def build_shared_prefix_edges(
    endpoints: list[RawEndpoint], idx: NodeIndex
) -> torch.Tensor:
    """Endpoints sharing ≥2 literal prefix segments → sibling_of edges.

    Parametric segments (e.g. {v}) don't count toward the threshold but
    don't break the LCP chain either.
    """
    # Collect unique (path_template, node_idx) pairs — one representative per path
    # For siblings we work at endpoint level (each (path, method) is a node)
    node_list: list[tuple[str, int]] = []
    for ep in endpoints:
        node_idx = idx.get((ep.path_template, ep.method))
        if node_idx is not None:
            node_list.append((ep.path_template, node_idx))

    src_list: list[int] = []
    dst_list: list[int] = []

    segs_cache = {path: _segments(path) for path, _ in node_list}

    for i in range(len(node_list)):
        path_a, idx_a = node_list[i]
        segs_a = segs_cache[path_a]
        for j in range(i + 1, len(node_list)):
            path_b, idx_b = node_list[j]
            if path_a == path_b:
                # Same path template — handled by crud_with, skip
                continue
            segs_b = segs_cache[path_b]

            # Compute LCP length and count literals within it
            lcp_len = 0
            for sa, sb in zip(segs_a, segs_b):
                if sa != sb:
                    break
                lcp_len += 1

            literal_count = sum(
                1 for seg in segs_a[:lcp_len] if not _is_param(seg)
            )

            if literal_count >= 2:
                src_list.extend([idx_a, idx_b])
                dst_list.extend([idx_b, idx_a])

    if not src_list:
        return _empty()
    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ---------------------------------------------------------------------------
# Edge 4: feeds — directed (response fields match request params)
# ---------------------------------------------------------------------------

def _stem(name: str) -> str:
    """Normalise a field/param name for fuzzy matching.

    Steps:
    1. Strip known ID suffixes: _id, Id, _ids  (case-sensitive patterns)
    2. Lowercase and remove underscores/hyphens
    3. Strip trailing 's' only if remaining name length > 4
    """
    # Strip known ID suffixes before lowercasing
    for suffix in ("_ids", "_id", "Id"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Lowercase and remove separators
    name = name.lower().replace("_", "").replace("-", "")

    # Strip trailing 's' only for longer names to avoid false positives
    if name.endswith("s") and len(name) > 4:
        name = name[:-1]

    return name


_COMPATIBLE_TYPES: set[frozenset[str]] = {
    frozenset({"string"}),
    frozenset({"integer"}),
    frozenset({"number"}),
    frozenset({"boolean"}),
    frozenset({"integer", "number"}),
}


def _types_compatible(t1: str, t2: str) -> bool:
    if t1 == t2:
        return True
    pair = frozenset({t1, t2})
    return pair in _COMPATIBLE_TYPES


def build_data_dependency_edges(
    endpoints: list[RawEndpoint],
    idx: NodeIndex,
    min_value_length: int = 8,
) -> torch.Tensor:
    """Producer → consumer edges based on response field / request param name matching.

    Matching logic:
    - Primary: _stem(field.name) == _stem(param.name) AND types compatible
    - No self-loops
    """
    src_list: list[int] = []
    dst_list: list[int] = []

    node_list: list[tuple[RawEndpoint, int]] = []
    for ep in endpoints:
        node_idx = idx.get((ep.path_template, ep.method))
        if node_idx is not None:
            node_list.append((ep, node_idx))

    # Precompute consumer param stems per node
    consumer_stems_map: dict[int, list[tuple[str, str]]] = {}
    for ep, nidx in node_list:
        params = list(ep.query_params) + list(ep.header_params) + list(ep.request_body_fields)
        consumer_stems_map[nidx] = [(_stem(p.name), p.type) for p in params]

    for i, (prod_ep, prod_idx) in enumerate(node_list):
        if not prod_ep.response_body_fields:
            continue

        prod_stems: dict[str, str] = {_stem(f.name): f.type for f in prod_ep.response_body_fields}

        for j, (cons_ep, cons_idx) in enumerate(node_list):
            if prod_idx == cons_idx:
                continue

            for p_stem, p_type in consumer_stems_map[cons_idx]:
                if p_stem in prod_stems and _types_compatible(prod_stems[p_stem], p_type):
                    src_list.append(prod_idx)
                    dst_list.append(cons_idx)
                    break

    if not src_list:
        return _empty()
    return torch.tensor([src_list, dst_list], dtype=torch.long)

