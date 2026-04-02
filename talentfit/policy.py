from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeaturePolicy:
    """
    Bias guardrails:
    - Location should not influence fit by default; treat it as a filter-only field.
    """

    use_location_in_scoring: bool = False

