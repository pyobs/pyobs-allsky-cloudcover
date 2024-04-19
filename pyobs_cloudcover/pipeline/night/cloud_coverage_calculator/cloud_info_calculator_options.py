from __future__ import annotations

from typing import Dict, Any


class CloudInfoCalculatorOptions(object):
    def __init__(self, cloud_threshold: float, zenith_radius: float) -> None:
        self.cloud_threshold = cloud_threshold
        self.altitude_limit = 90 - zenith_radius

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> CloudInfoCalculatorOptions:
        return CloudInfoCalculatorOptions(**options)
