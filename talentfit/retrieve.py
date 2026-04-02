from __future__ import annotations

import numpy as np


def cosine_topk(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, scores) for top-k cosine similarity.

    Assumes vectors are already L2-normalized (so dot product == cosine).
    """
    if query_vec.ndim != 1:
        raise ValueError("query_vec must be shape (d,)")
    if doc_matrix.ndim != 2:
        raise ValueError("doc_matrix must be shape (n, d)")
    if doc_matrix.shape[1] != query_vec.shape[0]:
        raise ValueError("dimension mismatch")

    scores = doc_matrix @ query_vec  # (n,)
    k = min(int(k), scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

