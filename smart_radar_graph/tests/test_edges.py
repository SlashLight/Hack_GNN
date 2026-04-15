import torch
import pytest
from smart_radar_graph.schema import RawEndpoint, ParamInfo, FieldInfo
from smart_radar_graph.edges import (
    build_resource_hierarchy_edges, build_crud_edges,
    build_shared_prefix_edges, build_data_dependency_edges,
)


def _ep(path="/api/users", method="GET", **kw) -> RawEndpoint:
    return RawEndpoint(path_template=path, method=method, **kw)

def _idx(endpoints):
    return {(e.path_template, e.method): i for i, e in enumerate(endpoints)}


class TestResourceHierarchy:
    def test_parent_child(self):
        eps = [_ep("/api/users"), _ep("/api/users/{id}")]
        edges = build_resource_hierarchy_edges(eps, _idx(eps))
        assert edges.shape[1] >= 1

    def test_no_parent(self):
        eps = [_ep("/api/users"), _ep("/api/posts")]
        edges = build_resource_hierarchy_edges(eps, _idx(eps))
        assert edges.shape[1] == 0

    def test_multi_method_parent(self):
        eps = [_ep("/api/users", "GET"), _ep("/api/users", "POST"), _ep("/api/users/{id}", "GET")]
        edges = build_resource_hierarchy_edges(eps, _idx(eps))
        assert edges.shape[1] == 2  # both parent methods → child


class TestCrudEdges:
    def test_crud_pair(self):
        eps = [_ep("/api/users", "GET"), _ep("/api/users", "POST")]
        edges = build_crud_edges(eps, _idx(eps))
        assert edges.shape[1] == 2  # both directions

    def test_no_crud_single_method(self):
        edges = build_crud_edges([_ep("/api/users", "GET")], _idx([_ep("/api/users", "GET")]))
        assert edges.shape[1] == 0

    def test_crud_clique_three(self):
        eps = [_ep("/api/users", m) for m in ("GET", "POST", "DELETE")]
        edges = build_crud_edges(eps, _idx(eps))
        assert edges.shape[1] == 6  # 3-clique both directions


class TestSharedPrefix:
    def test_two_literal_shared(self):
        eps = [_ep("/api/v1/users"), _ep("/api/v1/posts")]
        edges = build_shared_prefix_edges(eps, _idx(eps))
        assert edges.shape[1] == 2

    def test_one_literal_no_edge(self):
        eps = [_ep("/api/users"), _ep("/api/posts")]
        edges = build_shared_prefix_edges(eps, _idx(eps))
        assert edges.shape[1] == 0

    def test_parametric_dont_count(self):
        eps = [_ep("/api/{v}/users"), _ep("/api/{v}/posts")]
        edges = build_shared_prefix_edges(eps, _idx(eps))
        assert edges.shape[1] == 0  # only 1 literal (api)


class TestDataDependency:
    def test_field_name_match(self):
        eps = [
            _ep("/api/users", "GET", response_body_fields=[FieldInfo("user_id", "integer", 0)]),
            _ep("/api/orders", "POST", query_params=[ParamInfo("userId", "integer")]),
        ]
        edges = build_data_dependency_edges(eps, _idx(eps))
        assert edges.shape[1] >= 1
        assert edges[0, 0].item() == 0  # producer → consumer

    def test_no_match_incompatible_types(self):
        eps = [
            _ep("/api/users", "GET", response_body_fields=[FieldInfo("count", "string", 0)]),
            _ep("/api/orders", "POST", query_params=[ParamInfo("count", "integer")]),
        ]
        edges = build_data_dependency_edges(eps, _idx(eps))
        assert edges.shape[1] == 0

    def test_no_self_loop(self):
        eps = [_ep("/api/users", "GET",
            response_body_fields=[FieldInfo("id", "integer", 0)],
            query_params=[ParamInfo("id", "integer")])]
        edges = build_data_dependency_edges(eps, _idx(eps))
        assert edges.shape[1] == 0

    def test_stem_no_false_positive(self):
        """'status' in response should NOT match 'statu' anywhere."""
        eps = [
            _ep("/api/health", "GET", response_body_fields=[FieldInfo("status", "string", 0)]),
            _ep("/api/users", "GET", query_params=[ParamInfo("status_code", "string")]),
        ]
        edges = build_data_dependency_edges(eps, _idx(eps))
        # 'status' stems to 'status' (not 'statu'), 'status_code' stems to 'statuscode'
        # These should NOT match
        assert edges.shape[1] == 0


