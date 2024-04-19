from typing import Union, Tuple

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model.world_model import WorldModel


class SimpleModel(WorldModel):
    def __init__(self, a0: float, F: float, R: float, c_x: float, c_y: float) -> None:
        self._a0 = a0
        self._F = F
        self._R = R
        self._cx, self._cy = c_x, c_y

    def pix_to_altaz(self, x: Union[npt.NDArray[np.float_], float], y: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        az = self._a0 + np.arctan2((y - self._cy), (x - self._cx))
        r = np.sqrt((x - self._cx) ** 2 + (y - self._cy) ** 2)
        z = self._F * np.arcsin(r / self._R)
        alt = np.pi / 2 - z

        return alt, az

    def altaz_to_pix(self, alt: Union[npt.NDArray[np.float_], float], az: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        z = np.pi / 2 - alt
        r = self._R * np.sin(z / self._F)
        x = self._cx + r * np.cos(az - self._a0)
        y = self._cy + r * np.sin(az - self._a0)

        return x, y
