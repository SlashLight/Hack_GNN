import torch
import pytest
from torch_geometric.data import HeteroData
from smart_radar_graph import build_graph, extract_from_spec, RawEndpoint, ParamInfo, FieldInfo

CRAPI_LIKE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "crAPI-like", "version": "1.0"},
    "paths": {
        "/api/auth/login": {
            "post": {
                "security": [],
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {
                        "email": {"type": "string"}, "password": {"type": "string"}
                    }
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {
                        "token": {"type": "string"}, "user_id": {"type": "integer"}
                    }
                }}}}}
            }
        },
        "/api/users": {
            "get": {
                "parameters": [{"name": "page", "in": "query", "schema": {"type": "integer"}}],
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "array", "items": {"type": "object", "properties": {
                        "id": {"type": "integer"}, "name": {"type": "string"}, "email": {"type": "string"}
                    }}
                }}}}}
            }
        },
        "/api/users/{id}": {
            "get": {
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "integer"}, "required": True}],
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {
                        "id": {"type": "integer"}, "name": {"type": "string"},
                        "email": {"type": "string"}, "role": {"type": "string"}
                    }
                }}}}}
            },
            "put": {
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "integer"}, "required": True}],
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}}
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}
            }
        },
        "/api/users/{id}/posts": {
            "get": {
                "parameters": [
                    {"name": "id", "in": "path", "schema": {"type": "integer"}, "required": True},
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                ],
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "array", "items": {"type": "object", "properties": {
                        "post_id": {"type": "integer"}, "title": {"type": "string"}
                    }}
                }}}}}
            }
        },
        "/api/admin/users": {
            "get": {
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "array", "items": {"type": "object"}
                }}}}}
            }
        },
        "/api/health": {
            "get": {
                "security": [],
                "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}
            }
        }
    },
    "components": {"securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}},
    "security": [{"bearerAuth": []}],
}


class TestEndToEnd:
    def test_full_pipeline(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="crapi-like", source_mode="spec")
        assert isinstance(graph, HeteroData)
        assert graph['endpoint'].x.shape[1] == 35
        assert graph.api_name == "crapi-like"

    def test_feature_dim_invariant(self):
        eps = extract_from_spec(CRAPI_LIKE_SPEC)
        graph = build_graph(eps, api_name="t")
        # Число узлов может быть >= len(eps) если сработал virtual root fallback
        assert graph['endpoint'].x.shape[0] >= len(eps)
        assert graph['endpoint'].x.shape[1] == 35

    def test_no_nan(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        assert not torch.any(torch.isnan(graph['endpoint'].x))
        assert not torch.any(torch.isinf(graph['endpoint'].x))

    def test_edge_types_exist(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        actual = {et[1] for et in graph.edge_types}
        assert {'parent_of', 'crud_with', 'sibling_of', 'feeds'}.issubset(actual)

    def test_has_hierarchy_edges(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        assert graph['endpoint', 'parent_of', 'endpoint'].edge_index.shape[1] > 0

    def test_has_crud_edges(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        assert graph['endpoint', 'crud_with', 'endpoint'].edge_index.shape[1] > 0

    def test_has_data_dep(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        assert graph['endpoint', 'feeds', 'endpoint'].edge_index.shape[1] > 0

    def test_sensitive_on_login(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        login_idx = graph['endpoint'].path.index("/api/auth/login")
        assert graph['endpoint'].x[login_idx, 30] == 1.0

    def test_to_homogeneous(self):
        graph = build_graph(extract_from_spec(CRAPI_LIKE_SPEC), api_name="t")
        homo = graph.to_homogeneous()
        assert homo.x.shape[1] == 35
        assert hasattr(homo, 'edge_type')


class TestMinimal:
    def test_single_endpoint(self):
        graph = build_graph([RawEndpoint("/api/health", "GET")], api_name="min")
        assert graph['endpoint'].x.shape == (1, 35)

    def test_unconnected_gets_fallback(self):
        eps = [RawEndpoint("/a", "GET", auth_type="bearer-oauth", auth_required=True),
               RawEndpoint("/b", "POST", auth_type="none", auth_required=False)]
        graph = build_graph(eps, api_name="t")
        assert graph.connectivity_fallback_used is True
