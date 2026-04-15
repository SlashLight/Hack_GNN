"""Graph Builder — orchestrator for the Smart Radar heterogeneous graph.

Produces a PyG HeteroData object from a list of RawEndpoint objects.

Flow:
  1. Sort endpoints canonically
  2. Build node index
  3. Extract static features [N, 32]
  4. Build 4 edge types
  5. Compute structural features [N, 3] from edges (indices 32-34)
  6. Concatenate → [N, 35]
  7. Z-score normalize continuous features
  8. Connectivity check + fallbacks
  9. Assemble HeteroData
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from smart_radar_graph.schema import RawEndpoint
from smart_radar_graph.features import extract_static_features, STATIC_DIM
from smart_radar_graph.edges import (
    build_resource_hierarchy_edges,
    build_crud_edges,
    build_shared_prefix_edges,
    build_data_dependency_edges,
)
from smart_radar_graph.normalize import normalize_features, TOTAL_DIM
from smart_radar_graph.connectivity import check_connectivity, add_virtual_root


def _build_structural_features(
    crud_edges: torch.Tensor,
    feeds_edges: torch.Tensor,
    num_nodes: int,
) -> np.ndarray:
    """Compute structural features [N, 3]:

    Index 0 (→ global 32): num_crud_neighbors = degree in crud_with / 2
    Index 1 (→ global 33): data_dep_in_degree (count of times node appears as dst in feeds)
    Index 2 (→ global 34): data_dep_out_degree (count of times node appears as src in feeds)
    """
    struct = np.zeros((num_nodes, 3), dtype=np.float32)

    # --- Index 32: num_crud_neighbors ---
    # crud_with stores both directions, so raw degree = 2 * actual neighbours
    if crud_edges.numel() > 0:
        src = crud_edges[0]  # shape [E]
        counts = torch.zeros(num_nodes, dtype=torch.float32)
        counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        struct[:, 0] = (counts / 2.0).numpy()

    # --- Index 33: data_dep_in_degree ---
    # --- Index 34: data_dep_out_degree ---
    if feeds_edges.numel() > 0:
        feeds_src = feeds_edges[0]  # shape [E]
        feeds_dst = feeds_edges[1]

        out_counts = torch.zeros(num_nodes, dtype=torch.float32)
        out_counts.scatter_add_(0, feeds_src, torch.ones_like(feeds_src, dtype=torch.float32))

        in_counts = torch.zeros(num_nodes, dtype=torch.float32)
        in_counts.scatter_add_(0, feeds_dst, torch.ones_like(feeds_dst, dtype=torch.float32))

        struct[:, 1] = in_counts.numpy()
        struct[:, 2] = out_counts.numpy()

    return struct


def build_graph(
    endpoints: list[RawEndpoint],
    api_name: str = "",
    source_mode: str = "spec",
) -> HeteroData:
    """Build a HeteroData graph from a list of RawEndpoint objects.

    Args:
        endpoints: List of RawEndpoint objects describing API endpoints.
        api_name: Name of the API (stored as graph metadata).
        source_mode: "spec" or "traffic" (stored as graph metadata).

    Returns:
        PyG HeteroData with node features [N, 35] and 5 (or 6) edge types.
    """
    # ------------------------------------------------------------------
    # Step 1: Sort endpoints canonically
    # ------------------------------------------------------------------
    if not endpoints:
        raise ValueError("build_graph requires at least one endpoint")

    sorted_eps = sorted(endpoints, key=lambda e: (e.path_template, e.method))
    N = len(sorted_eps)

    # ------------------------------------------------------------------
    # Step 2: Build node index  {(path_template, method) → i}
    # ------------------------------------------------------------------
    node_idx: dict[tuple[str, str], int] = {
        (ep.path_template, ep.method): i for i, ep in enumerate(sorted_eps)
    }

    # ------------------------------------------------------------------
    # Step 3: Extract static features [N, 32]
    # ------------------------------------------------------------------
    static_feats = extract_static_features(sorted_eps)  # [N, 32]
    assert static_feats.shape == (N, STATIC_DIM), (
        f"Expected static features shape ({N}, {STATIC_DIM}), got {static_feats.shape}"
    )

    # ------------------------------------------------------------------
    # Step 4: Build all 4 edge types
    # ------------------------------------------------------------------
    hierarchy_edges = build_resource_hierarchy_edges(sorted_eps, node_idx)
    crud_edges = build_crud_edges(sorted_eps, node_idx)
    sibling_edges = build_shared_prefix_edges(sorted_eps, node_idx)
    feeds_edges = build_data_dependency_edges(sorted_eps, node_idx)

    edge_dict: dict[str, torch.Tensor] = {
        "parent_of": hierarchy_edges,
        "crud_with": crud_edges,
        "sibling_of": sibling_edges,
        "feeds": feeds_edges,
    }

    # ------------------------------------------------------------------
    # Step 5: Compute structural features [N, 3]
    # ------------------------------------------------------------------
    struct_feats = _build_structural_features(crud_edges, feeds_edges, N)  # [N, 3]

    # ------------------------------------------------------------------
    # Step 6: Concatenate → [N, 35]
    # ------------------------------------------------------------------
    features_np = np.concatenate([static_feats, struct_feats], axis=1)  # [N, 35]
    assert features_np.shape == (N, TOTAL_DIM), (
        f"Expected feature matrix shape ({N}, {TOTAL_DIM}), got {features_np.shape}"
    )

    # ------------------------------------------------------------------
    # Step 7: Normalize continuous features (z-score)
    # ------------------------------------------------------------------
    features_np = normalize_features(features_np)

    # ------------------------------------------------------------------
    # Step 8: Connectivity check with ordered fallbacks
    # ------------------------------------------------------------------
    connectivity = check_connectivity(edge_dict, N)
    num_components_before = connectivity.num_components
    fallback_used = False

    if not connectivity.is_connected:
        # Fallback 1: rebuild data_dependency edges with min_value_length=6
        feeds_edges_f1 = build_data_dependency_edges(sorted_eps, node_idx, min_value_length=6)
        edge_dict_f1 = dict(edge_dict)
        edge_dict_f1["feeds"] = feeds_edges_f1

        connectivity_f1 = check_connectivity(edge_dict_f1, N)

        if connectivity_f1.is_connected:
            # Fallback 1 resolved disconnection — use updated edges + recompute features
            edge_dict = edge_dict_f1
            feeds_edges = feeds_edges_f1
            struct_feats = _build_structural_features(crud_edges, feeds_edges, N)
            features_np = normalize_features(np.concatenate([static_feats, struct_feats], axis=1))
            fallback_used = True
        else:
            # Fallback 2: add virtual root node
            edge_dict_f2, N_new = add_virtual_root(edge_dict_f1, N)

            connectivity_f2 = check_connectivity(edge_dict_f2, N_new)
            assert connectivity_f2.is_connected, (
                "Graph is still disconnected after adding virtual root — this should never happen."
            )

            # Extend feature matrix with a zero row for the virtual root
            zero_row = np.zeros((1, TOTAL_DIM), dtype=np.float32)
            features_np = np.vstack([features_np, zero_row])  # [N+1, 35]

            # Add virtual root to path/method lists
            sorted_eps = list(sorted_eps)  # make mutable copy
            sorted_eps.append(RawEndpoint(path_template="__virtual_root__", method="VIRTUAL"))

            edge_dict = edge_dict_f2
            N = N_new
            fallback_used = True

    # ------------------------------------------------------------------
    # Step 9: Assemble HeteroData
    # ------------------------------------------------------------------
    data = HeteroData()

    data['endpoint'].x = torch.tensor(features_np, dtype=torch.float32)
    data['endpoint'].path = [ep.path_template for ep in sorted_eps]
    data['endpoint'].method = [ep.method for ep in sorted_eps]

    for edge_type, edge_tensor in edge_dict.items():
        data['endpoint', edge_type, 'endpoint'].edge_index = edge_tensor

    # Graph-level metadata
    data.api_name = api_name
    data.source_mode = source_mode
    data.num_components_before_fallback = num_components_before
    data.connectivity_fallback_used = fallback_used
    data.feature_schema_version = "1.0"

    return data
