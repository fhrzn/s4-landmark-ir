import numpy as np

from src.utils import haversine


def precision_k(
    gt_gps: np.ndarray, ret_gps: np.ndarray, k: int = 10, min_dist: int = 50
):
    """Out of the top-K results, how many are actually outside the forbidden radius?"""
    distances = haversine(gt_gps, ret_gps).T
    if len(distances.shape) >= 2:
        return np.mean(distances[:, :k] >= min_dist).item()
    return np.mean(distances[:k] >= min_dist).item()
