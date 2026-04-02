from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CutoffPolicy:
    """
    Simple default cutoff policy intended to generalize across roles.

    Rationale:
    - We have weak/no ground-truth labels, so we avoid overfitting a threshold.
    - We use robust quantile + absolute floor to avoid returning junk.
    """

    min_fit: float = 0.60
    keep_quantile: float = 0.70  # keep top 30% by fit (within retrieved set)

    def decide(self, fit_scores: np.ndarray) -> float:
        fit = np.asarray(fit_scores, dtype=float)
        if fit.size == 0:
            return 1.0
        q = float(np.quantile(fit, self.keep_quantile))
        return max(self.min_fit, q)

