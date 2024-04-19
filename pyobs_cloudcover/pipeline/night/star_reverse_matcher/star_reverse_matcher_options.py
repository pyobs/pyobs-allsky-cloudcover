from __future__ import annotations

from typing import Dict, Any


class StarReverseMatcherOptions(object):
    def __init__(self, sigma_threshold: float = 3.0, window_size: int = 6):
        self.sigma_threshold = sigma_threshold
        self.window_size = window_size

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> StarReverseMatcherOptions:
        return StarReverseMatcherOptions(**options)
