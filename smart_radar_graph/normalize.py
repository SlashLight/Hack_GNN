from __future__ import annotations
import numpy as np

TOTAL_DIM = 35
CONTINUOUS_INDICES = [5, 6, 10, 11, 17, 32, 33, 34]


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Per-graph z-score on continuous columns. Non-continuous unchanged. std=0 → leave as-is."""
    result = features.copy()
    for col_idx in CONTINUOUS_INDICES:
        if col_idx >= result.shape[1]:
            continue
        col = result[:, col_idx]
        std = col.std()
        if std > 1e-8:
            result[:, col_idx] = (col - col.mean()) / std
    return result
