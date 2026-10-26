from __future__ import annotations


class CloudMapGeneratorOptions:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    @classmethod
    def from_dict(cls, options: dict[str, float]) -> CloudMapGeneratorOptions:
        return cls(**options)
