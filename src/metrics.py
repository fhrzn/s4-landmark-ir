from typing import Dict

import numpy as np

from src.constant import LABELS2ID
from src.utils import haversine


def precision_k(
    gt_gpses: np.ndarray,
    gt_labels: np.ndarray,
    ref_meta: Dict,
    topk: int,
    min_dist: int,
):
    prec_k = 0
    for key, ref in ref_meta.items():
        idx = int(key)
        ref_gps = np.array([[r["LAT"], r["LON"]] for r in ref[:topk]])
        ref_labels = np.array([LABELS2ID[r["label"]] for r in ref[:topk]])

        relevance = ref_labels == gt_labels[idx]
        distances = haversine(gt_gpses[idx], ref_gps) >= min_dist
        prec_k += np.average(relevance * distances)

    prec_k /= len(ref_meta)
    return prec_k


def map_k(
    gt_gpses: np.ndarray,
    gt_labels: np.ndarray,
    ref_meta: Dict,
    topk: int,
    min_dist: int,
):
    map_k = 0
    for key, ref in ref_meta.items():
        idx = int(key)
        ref_gps = np.array([[r["LAT"], r["LON"]] for r in ref[:topk]])
        ref_labels = np.array([LABELS2ID[r["label"]] for r in ref[:topk]])

        relevance = ref_labels == gt_labels[idx]
        distances = haversine(gt_gpses[idx], ref_gps) >= min_dist

        rel_far = (relevance * distances)[:topk]
        prec_i = np.cumsum(rel_far) / np.arange(1, topk+1)
        hits = rel_far.sum()
        ap_k = 0.0 if hits == 0 else (np.average(prec_i * rel_far) / hits)

        map_k += ap_k

    map_k /= len(ref_meta)
    return map_k
