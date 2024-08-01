import functools
from typing import List, Optional, Tuple

import numpy as np
from cloudmap_rs import AltAzCoord

from pyobs_cloudcover.world_model import WorldModel


class AltAzMapGenerator:
    def __init__(self, model: WorldModel, limiting_altitude: float):
        self._model = model
        self._limiting_altitude = limiting_altitude

    @functools.lru_cache(maxsize=None)
    def __call__(self, image_height: int, image_width: int) -> List[List[Optional[AltAzCoord]]]:
        x = np.arange(0, image_width)
        y = np.arange(0, image_height)

        meshed_x, meshed_y = np.meshgrid(x, y)

        alt, az = self._model.pix_to_altaz(meshed_x, meshed_y)
        alt_az_coords = np.dstack((alt, az))

        return [
                [
                    self._alt_az_convert(coord)
                    for coord in row
                ] for row in alt_az_coords
            ]

    def _alt_az_convert(self, alt_az_coords: Tuple[float, float]) -> Optional[AltAzCoord]:
        if alt_az_coords[0] < np.deg2rad(self._limiting_altitude) or np.isnan(alt_az_coords[0]) or np.isnan(alt_az_coords[1]):
            return None
        else:
            return AltAzCoord(*alt_az_coords)
