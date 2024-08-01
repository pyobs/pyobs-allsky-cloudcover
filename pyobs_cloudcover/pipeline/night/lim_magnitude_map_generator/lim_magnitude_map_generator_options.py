from __future__ import annotations

from typing import Any, Dict


class LimMagnitudeMapGeneratorOptions(object):
    def __init__(self, radius: float = 5.0, integration_length: int = 1) -> None:
        self.radius = radius
        self.integration_length = integration_length

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> LimMagnitudeMapGeneratorOptions:
        return cls(**options)
