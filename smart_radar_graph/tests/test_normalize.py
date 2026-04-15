import numpy as np
import pytest
from smart_radar_graph.normalize import normalize_features, CONTINUOUS_INDICES, TOTAL_DIM


class TestConstants:
    def test_total_dim(self):
        assert TOTAL_DIM == 35

    def test_continuous_indices(self):
        assert CONTINUOUS_INDICES == [5, 6, 10, 11, 17, 32, 33, 34]


class TestNormalization:
    def test_continuous_columns_normalized(self):
        features = np.zeros((5, 35), dtype=np.float32)
        features[:, 5] = [1, 2, 3, 4, 5]
        result = normalize_features(features)
        assert abs(result[:, 5].mean()) < 1e-6

    def test_non_continuous_unchanged(self):
        features = np.zeros((3, 35), dtype=np.float32)
        features[:, 0] = [1, 0, 0]  # method one-hot
        features[:, 5] = [2, 4, 6]  # path_depth
        result = normalize_features(features)
        assert list(result[:, 0]) == [1, 0, 0]

    def test_zero_std_no_crash(self):
        features = np.zeros((3, 35), dtype=np.float32)
        features[:, 5] = [3, 3, 3]
        result = normalize_features(features)
        assert list(result[:, 5]) == [3, 3, 3]

    def test_single_node(self):
        features = np.zeros((1, 35), dtype=np.float32)
        features[0, 5] = 5.0
        result = normalize_features(features)
        assert result[0, 5] == 5.0

    def test_no_nan_no_inf(self):
        features = np.random.rand(10, 35).astype(np.float32)
        result = normalize_features(features)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_returns_copy(self):
        features = np.zeros((3, 35), dtype=np.float32)
        features[:, 5] = [1, 2, 3]
        original = features.copy()
        normalize_features(features)
        np.testing.assert_array_equal(features, original)
