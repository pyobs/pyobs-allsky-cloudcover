import functools

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model import WorldModel


class AltAzMapGenerator:
    def __init__(self, model: WorldModel, limiting_altitude: float):
        self._model = model
        self._limiting_altitude = limiting_altitude

    @functools.cache
    def __call__(
        self, image_height: int, image_width: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Altitude and azimuth (rad) of every pixel, and whether the pixel is on the sky above the altitude limit."""
        meshed_x, meshed_y = np.meshgrid(
            np.arange(image_width, dtype=np.float64), np.arange(image_height, dtype=np.float64)
        )

        with np.errstate(invalid="ignore"):
            alt, az = self._model.pix_to_altaz(meshed_x, meshed_y)
            alt, az = np.asarray(alt, dtype=np.float64), np.asarray(az, dtype=np.float64)
            valid = ~np.isnan(alt) & ~np.isnan(az) & (alt >= np.deg2rad(self._limiting_altitude))

        return alt, az, valid
