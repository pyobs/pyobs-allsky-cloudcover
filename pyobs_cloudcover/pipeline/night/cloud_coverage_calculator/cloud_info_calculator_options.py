from __future__ import annotations

from typing import Dict, Any


class CloudInfoCalculatorOptions(object):
    def __init__(self, altitude_limit: float) -> None:
        self.altitude_limit = altitude_limit

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> CloudInfoCalculatorOptions:
        return CloudInfoCalculatorOptions(**options)
