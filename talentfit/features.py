from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass(frozen=True)
class FeatureWeights:
    w_embed: float = 0.85
    w_lex: float = 0.10
    w_conn: float = 0.05


def build_features(
    *,
    embed_sim: np.ndarray,  # (n,)
    query_text: str,
    doc_texts: list[str],
    connections: np.ndarray,  # (n,) numeric
) -> dict[str, np.ndarray]:
    q_tokens = _tokenize(query_text)
    lex = np.array([jaccard(q_tokens, _tokenize(t)) for t in doc_texts], dtype=np.float32)

    # Connections: robust squash into 0..1
    conn = connections.astype(np.float32)
    conn = np.nan_to_num(conn, nan=0.0)
    conn_feat = np.log1p(conn) / np.log1p(500.0)  # cap-ish at 500+
    conn_feat = np.clip(conn_feat, 0.0, 1.0).astype(np.float32)

    return {
        "embed_sim": embed_sim.astype(np.float32),
        "lex_sim": lex,
        "conn": conn_feat,
    }


def score_fit(features: dict[str, np.ndarray], w: FeatureWeights) -> np.ndarray:
    raw = (
        w.w_embed * features["embed_sim"]
        + w.w_lex * features["lex_sim"]
        + w.w_conn * features["conn"]
    )

    # Calibrate into 0..1 with a smooth monotonic transform.
    # (Not a probability yet; later we will calibrate with feedback.)
    # Center around 0.4..0.6 typical cosine range for MiniLM.
    x = (raw - 0.35) / 0.15
    fit = 1.0 / (1.0 + np.exp(-x))
    return fit.astype(np.float32)

