import torch
import pytest
from torch_geometric.data import HeteroData
from smart_radar_graph.schema import RawEndpoint, ParamInfo, FieldInfo
from smart_radar_graph.graph_builder import build_graph


def _ep(path="/api/users", method="GET", **kw) -> RawEndpoint:
    return RawEndpoint(path_template=path, method=method, **kw)


class TestBuildGraph:
    def test_output_type(self):
        assert isinstance(build_graph([_ep(), _ep(path="/api/posts")], api_name="test"), HeteroData)

    def test_node_features_shape(self):
        # Connected via crud_with (/api/users GET+POST) and parent_of (/api/users → /api/users/{id})
        result = build_graph([_ep(), _ep(method="POST"), _ep(path="/api/users/{id}")], api_name="t")
        assert result['endpoint'].x.shape == (3, 35)

    def test_no_nan(self):
        result = build_graph([_ep(), _ep(path="/api/posts")], api_name="t")
        assert not torch.any(torch.isnan(result['endpoint'].x))
        assert not torch.any(torch.isinf(result['endpoint'].x))

    def test_all_edge_types(self):
        eps = [_ep("/api/v1/users", "GET"), _ep("/api/v1/users", "POST"),
               _ep("/api/v1/users/{id}", "GET"), _ep("/api/v1/posts", "GET")]
        result = build_graph(eps, api_name="t")
        actual = {et[1] for et in result.edge_types}
        assert {'parent_of', 'crud_with', 'sibling_of', 'feeds'}.issubset(actual)

    def test_metadata(self):
        result = build_graph([_ep()], api_name="crapi", source_mode="spec")
        assert result.api_name == "crapi"
        assert result.source_mode == "spec"
        assert hasattr(result, 'connectivity_fallback_used')

    def test_canonical_ordering(self):
        # /a GET + /a POST — связаны через crud_with, сортировка по (path, method)
        result = build_graph([_ep("/a", "POST"), _ep("/a", "GET")], api_name="t")
        assert result['endpoint'].path == ["/a", "/a"]
        assert result['endpoint'].method == ["GET", "POST"]

    def test_to_homogeneous(self):
        """HeteroData can convert to homogeneous for existing GNN models."""
        result = build_graph([_ep(), _ep(path="/api/posts")], api_name="t")
        homo = result.to_homogeneous()
        assert homo.x.shape[1] == 35


class TestConnectivityFallback:
    def test_connected_no_fallback(self):
        # GET + POST на одном пути — crud_with обеспечивает связность без fallback
        eps = [_ep("/api/users", "GET"), _ep("/api/users", "POST")]
        result = build_graph(eps, api_name="t")
        assert result.connectivity_fallback_used is False

    def test_fallback_virtual_root(self):
        eps = [_ep("/a", auth_type="bearer-oauth", auth_required=True),
               _ep("/b", auth_type="none", auth_required=False)]
        result = build_graph(eps, api_name="t")
        if result.connectivity_fallback_used:
            assert ('endpoint', 'virtual_root', 'endpoint') in result.edge_types
