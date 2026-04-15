from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import torch


@dataclass
class ConnectivityResult:
    is_connected: bool
    num_components: int
    component_sizes: list[int]


def check_connectivity(edge_dict: dict[str, torch.Tensor], num_nodes: int) -> ConnectivityResult:
    if num_nodes <= 1:
        return ConnectivityResult(True, max(num_nodes, 0), [1] * num_nodes)
    adj: dict[int, set[int]] = defaultdict(set)
    for edges in edge_dict.values():
        if edges.numel() == 0:
            continue
        src_nodes = edges[0].tolist()
        dst_nodes = edges[1].tolist()
        for u, v in zip(src_nodes, dst_nodes):
            adj[u].add(v)
            adj[v].add(u)
    visited, components = set(), []
    for node in range(num_nodes):
        if node in visited:
            continue
        comp = []
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            queue.extend(adj[n] - visited)
        components.append(len(comp))
    return ConnectivityResult(len(components) == 1, len(components), sorted(components, reverse=True))


def add_virtual_root(edge_dict: dict[str, torch.Tensor], num_nodes: int) -> tuple[dict[str, torch.Tensor], int]:
    vr = num_nodes
    new_edges = dict(edge_dict)
    src = list(range(num_nodes)) + [vr] * num_nodes
    dst = [vr] * num_nodes + list(range(num_nodes))
    new_edges['virtual_root'] = torch.tensor([src, dst], dtype=torch.long)
    return new_edges, num_nodes + 1
