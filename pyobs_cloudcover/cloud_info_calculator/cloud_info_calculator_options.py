from __future__ import annotations

from typing import Dict, Any


class CloudInfoCalculatorOptions(object):
    def __init__(self, zenith_range: float) -> None:
        self.zenith_range = zenith_range

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> CloudInfoCalculatorOptions:
        return CloudInfoCalculatorOptions(**options)
