from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from preprocessor.models import ProcessedEvent
from graph.value_tracker import ValueTracker
from graph.vocabulary import UnifiedVocabulary

_UNIFIED_VOCAB_SIZE = 128
_FEAT_DIM = 13 + _UNIFIED_VOCAB_SIZE  # 141


class GraphBuilder:
    """Инкрементальное построение гомогенного графа endpoint-узлов.

    Рёбра одного типа:
    - data_dep: параметр текущего запроса совпал с leaf-значением из предыдущего ответа
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._value_tracker = ValueTracker()
        self._endpoint_events: dict[str, list[ProcessedEvent]] = {}
        self._vocabulary: UnifiedVocabulary | None = None

    def build_vocabulary(self) -> None:
        """Первый проход: построить unified vocabulary из накопленных событий."""
        self._vocabulary = UnifiedVocabulary()
        self._vocabulary.build_from_events(self._endpoint_events)

    def add_event(self, event: ProcessedEvent) -> None:
        eid = event.endpoint_id

        # 1. Создать/обновить узел
        if eid not in self._graph:
            self._graph.add_node(eid)
        self._endpoint_events.setdefault(eid, []).append(event)

        # 2. Data dependency edges
        deps = self._value_tracker.find_dependencies(eid, event.parameters)
        for src, dst, matched_value, param_name in deps:
            if self._graph.has_edge(src, dst):
                self._graph[src][dst].setdefault("types", set()).add("data_dep")
                self._graph[src][dst].setdefault("data_dep_values", set()).add(matched_value)
                self._graph[src][dst].setdefault("data_dep_params", set()).add(param_name)
            else:
                self._graph.add_edge(
                    src, dst,
                    types={"data_dep"},
                    data_dep_values={matched_value},
                    data_dep_params={param_name},
                )

        # 3. Record response values + path param values from request URL.
        # Только path params: original_value — реальный ID ресурса (e.g. "42").
        # Query param names (e.g. "limit") не записываем — это имена, не значения.
        self._value_tracker.record_response(eid, event.response_values)
        url_values: dict[str, str] = {}
        for p in event.parameters:
            if p.location == "path" and p.original_value:
                url_values[p.original_value] = p.original_value
        if url_values:
            self._value_tracker.record_response(eid, url_values)

    def to_pyg_data(self) -> Data:
        if self._vocabulary is None:
            self.build_vocabulary()

        nodes = sorted(self._graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        vocab = self._vocabulary

        x = np.zeros((N, _FEAT_DIM), dtype=np.float32)

        METHODS = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "PATCH": 4}

        for i, node in enumerate(nodes):
            events = self._endpoint_events.get(node, [])
            if not events:
                continue
            total = len(events)

            # 0-4: method one-hot (из первого события)
            x[i, METHODS.get(events[0].method, 0)] = 1.0

            # 5: has_dynamic_segment
            x[i, 5] = 1.0 if any(e.has_dynamic_segment for e in events) else 0.0

            # 6-8: auth_dist [anon, cookie, token]
            # token покрывает Bearer + Basic + ApiKey, чтобы сумма всегда = 1.0
            anon = sum(1 for e in events if e.auth_type is None)
            cookie = sum(1 for e in events if e.auth_type == "Cookie")
            token = sum(1 for e in events if e.auth_type is not None and e.auth_type != "Cookie")
            x[i, 6] = anon / total
            x[i, 7] = cookie / total
            x[i, 8] = token / total

            # 9: mean_entropy по всем param.value_entropy
            entropies = [p.value_entropy for e in events for p in e.parameters]
            x[i, 9] = sum(entropies) / len(entropies) if entropies else 0.0

            # 10-12: status_dist [2xx, 4xx, 5xx]
            x[i, 10] = sum(1 for e in events if 200 <= e.status_code < 300) / total
            x[i, 11] = sum(1 for e in events if 400 <= e.status_code < 500) / total
            x[i, 12] = sum(1 for e in events if 500 <= e.status_code < 600) / total

            # 13..140: unified vocabulary (128 бина) — param names + response keys
            seen_params = {p.name for e in events for p in e.parameters}
            seen_resp = {k for e in events for k in e.response_keys}
            all_names = seen_params | seen_resp
            for name in all_names:
                x[i, 13 + vocab.get_index(name)] = 1.0

        # Edges
        src_list: list[int] = []
        dst_list: list[int] = []
        edge_attrs: list[list[float]] = []
        for u, v, data in self._graph.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue
            types = data.get("types", set())
            src_list.append(node_to_idx[u])
            dst_list.append(node_to_idx[v])

            edge_feat = [0.0] * (1 + _UNIFIED_VOCAB_SIZE)
            edge_feat[0] = 1.0 if "data_dep" in types else 0.0
            for pname in data.get("data_dep_params", set()):
                edge_feat[1 + vocab.get_index(pname)] = 1.0

            edge_attrs.append(edge_feat)

        edge_feat_dim = 1 + _UNIFIED_VOCAB_SIZE  # 129
        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, edge_feat_dim), dtype=torch.float32)

        return Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def to_networkx(self) -> nx.DiGraph:
        return self._graph.copy()
