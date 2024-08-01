import functools
from typing import List

import numpy as np
from cloudmap_rs import AltAzCoord
from pyobs.utils.grids import SphericalGrid


class SphericalAltAzGenerator:
    def __init__(self, point_number: int, limiting_altitude: float):
        self._point_number = point_number
        self._limiting_altitude = limiting_altitude

    @functools.lru_cache(maxsize=None)
    def __call__(self) -> List[AltAzCoord]:
        points = np.deg2rad(SphericalGrid.equidistributed(self._point_number))
        coords = [AltAzCoord(y, x) for x, y in points]
        filtered_coords = [coord for coord in coords if coord.alt >= np.deg2rad(self._limiting_altitude)]

        return filtered_coords
