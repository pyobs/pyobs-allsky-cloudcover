from __future__ import annotations

from typing import Any


class AltAzGridOptions:
    def __init__(self, point_number: int, limiting_altitude: float):
        self.limiting_altitude = limiting_altitude
        self.point_number = point_number

    @staticmethod
    def from_dict(options: dict[str, Any]) -> AltAzGridOptions:
        return AltAzGridOptions(options["point_number"], options["limiting_altitude"])
