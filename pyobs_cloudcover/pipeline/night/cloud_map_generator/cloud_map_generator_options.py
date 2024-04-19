from __future__ import annotations

from typing import Any, Dict


class CloudMapGeneratorOptions(object):
    def __init__(self, radius: float = 5.0) -> None:
        self.radius = radius

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> CloudMapGeneratorOptions:
        return cls(**options)
