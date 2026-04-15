from __future__ import annotations

import networkx as nx
from pyvis.network import Network


def draw_graph(G: nx.DiGraph, output: str = "graph.html") -> str:
    """Визуализирует граф эндпоинтов через pyvis.

    Цветовое кодирование рёбер:
    - data_dep: красный (#cc0000), толщина 3 — потенциальный IDOR
    - temporal:  серый (#888888), толщина 1 — последовательность запросов
    """
    net = Network(directed=True, height="750px", width="100%")
    for node in G.nodes():
        net.add_node(node, label=node, title=node)
    for u, v, data in G.edges(data=True):
        types = data.get("types", set())
        color = "#cc0000" if "data_dep" in types else "#888888"
        width = 3 if "data_dep" in types else 1
        # Подпись ребра
        parts: list[str] = []
        if "temporal" in types:
            count = data.get("temporal_count", 1)
            parts.append(f"×{count}" if count > 1 else "seq")
        if "data_dep" in types:
            vals = data.get("data_dep_values", set())
            parts.append(",".join(sorted(vals)) if vals else "dep")
        label = " | ".join(parts)
        net.add_edge(u, v, color=color, width=width, label=label, title=label)
    net.write_html(output)
    return output
