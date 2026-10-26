from __future__ import annotations

from typing import Any


class AltAzGridOptions:
    def __init__(self, point_number: int = 100000, limiting_altitude: float = 30.0):
        self.limiting_altitude = float(limiting_altitude)
        self.point_number = int(float(point_number))

    @staticmethod
    def from_dict(options: dict[str, Any]) -> AltAzGridOptions:
        return AltAzGridOptions(**options)
