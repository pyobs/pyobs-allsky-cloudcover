from __future__ import annotations

from typing import Any


class StarReverseMatcherOptions:
    def __init__(
        self, sigma_threshold: float = 3.0, distance: float = 4.0, median_limit: float = 3e8, window_size: int = 6
    ):
        self.sigma_threshold = sigma_threshold
        self.distance = distance
        self.median_limit = median_limit
        self.window_size = window_size

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> StarReverseMatcherOptions:
        return StarReverseMatcherOptions(**options)
