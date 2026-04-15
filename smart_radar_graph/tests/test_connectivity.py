import torch
import pytest
from smart_radar_graph.connectivity import check_connectivity, add_virtual_root, ConnectivityResult


class TestConnectivity:
    def test_connected(self):
        edges = {'parent_of': torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}
        result = check_connectivity(edges, num_nodes=3)
        assert result.is_connected and result.num_components == 1

    def test_disconnected(self):
        edges = {'parent_of': torch.tensor([[0], [1]], dtype=torch.long)}
        result = check_connectivity(edges, num_nodes=3)
        assert not result.is_connected and result.num_components == 2

    def test_empty_edges(self):
        result = check_connectivity({}, num_nodes=3)
        assert not result.is_connected and result.num_components == 3

    def test_single_node(self):
        result = check_connectivity({}, num_nodes=1)
        assert result.is_connected


class TestVirtualRoot:
    def test_adds_edges(self):
        edges = {'parent_of': torch.tensor([[0], [1]], dtype=torch.long)}
        new_edges, new_n = add_virtual_root(edges, num_nodes=3)
        assert 'virtual_root' in new_edges
        assert new_n == 4
        assert new_edges['virtual_root'].shape[1] == 6  # 3 nodes * 2 directions

    def test_original_preserved(self):
        edges = {'parent_of': torch.tensor([[0], [1]], dtype=torch.long)}
        new_edges, _ = add_virtual_root(edges, num_nodes=2)
        assert torch.equal(new_edges['parent_of'], edges['parent_of'])
