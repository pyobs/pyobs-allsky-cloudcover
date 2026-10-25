import functools
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from cloudmap_rs import AltAzCoord


def fibonacci_sphere(point_number: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Altitude and azimuth (in rad) of ``point_number`` points evenly distributed over the whole sphere.

    Uses the Fibonacci lattice: equal-area latitude bands, golden-angle azimuth steps.
    """
    i = np.arange(point_number) + 0.5
    alt = np.arcsin(1 - 2 * i / point_number)
    az = np.mod(np.pi * (1 + np.sqrt(5)) * i, 2 * np.pi)
    return alt, az


class SphericalAltAzGenerator:
    def __init__(self, point_number: int, limiting_altitude: float):
        self._point_number = point_number
        self._limiting_altitude = limiting_altitude

    @functools.lru_cache(maxsize=None)
    def __call__(self) -> List[AltAzCoord]:
        alt, az = fibonacci_sphere(self._point_number)
        above = alt >= np.deg2rad(self._limiting_altitude)

        return [AltAzCoord(a, z) for a, z in zip(alt[above], az[above])]
