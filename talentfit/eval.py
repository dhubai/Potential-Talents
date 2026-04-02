from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2**rel - 1) * discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)
    best = np.sort(rel)[::-1]
    ideal = dcg_at_k(best, k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(rel, k) / ideal


def average_precision_at_k(binary_relevances: np.ndarray, k: int) -> float:
    rel = (np.asarray(binary_relevances)[:k] > 0).astype(int)
    if rel.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def map_at_k(binary_relevances: np.ndarray, k: int) -> float:
    # Single-query MAP is AP; for multi-query we'd average.
    return average_precision_at_k(binary_relevances, k)

