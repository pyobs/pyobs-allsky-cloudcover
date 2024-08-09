from __future__ import annotations
from typing import Dict


class CloudMapGeneratorOptions(object):
    def __init__(self, threshold: float):
        self.threshold = threshold

    @classmethod
    def from_dict(cls, options: Dict[str, float]) -> CloudMapGeneratorOptions:
        return cls(**options)
