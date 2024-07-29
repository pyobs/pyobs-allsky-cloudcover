from typing import Optional, cast

import numpy as np
from cloudmap_rs import AltAzCoord, SkyPixelQuery


class ZenithCloudCoverageCalculator(object):
    def __init__(self, altitude: float) -> None:
        self._altitude = altitude

    def __call__(self, sky_query: SkyPixelQuery) -> Optional[float]:
        radius = np.pi/2 - np.deg2rad(self._altitude)
        return cast(Optional[float], sky_query.query_radius(AltAzCoord(np.pi/2, 0), radius))


