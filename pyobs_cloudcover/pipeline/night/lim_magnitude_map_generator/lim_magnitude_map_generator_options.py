from __future__ import annotations

from typing import Any


class LimMagnitudeMapGeneratorOptions:
    """``estimator``: ``robust`` (best separating cut) or ``border`` (1.x: faintest seen vs brightest unseen)."""

    def __init__(self, radius: float = 7.0, integration_length: int = 1, estimator: str = "border") -> None:
        if estimator not in ("robust", "border"):
            raise ValueError(f"Unknown estimator {estimator}")
        self.radius = float(radius)
        self.integration_length = int(integration_length)
        self.estimator = estimator

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> LimMagnitudeMapGeneratorOptions:
        return cls(**options)
