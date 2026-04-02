from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BiasReport:
    n_total: int
    n_kept: int
    kept_rate: float


def simple_impact_report(ranked: pd.DataFrame) -> BiasReport:
    """
    Minimal impact report: how aggressive the cutoff is.
    (We don't have protected attributes; location is treated as filter-only.)
    """
    n_total = int(len(ranked))
    n_kept = int(ranked.get("keep", pd.Series([1] * len(ranked))).sum())
    kept_rate = float(n_kept / n_total) if n_total else 0.0
    return BiasReport(n_total=n_total, n_kept=n_kept, kept_rate=kept_rate)

