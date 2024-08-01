from typing import Optional

import numpy as np
import numpy.typing as npt


class CoverageChangeCalculator(object):
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._last_map: Optional[npt.NDArray[np.float_]] = None

    def __call__(self, limiting_mag_map: npt.NDArray[np.float_]) -> Optional[float]:
        if self._last_map is None:
            self._last_map = limiting_mag_map
            return None

        image_filter = np.isnan(limiting_mag_map) | np.isnan(self._last_map)
        clouds = (limiting_mag_map[~image_filter] <= self._threshold)
        last_clouds = (self._last_map[~image_filter] <= self._threshold)

        pixel_count = np.sum(~image_filter)

        changed_pixels = np.logical_xor(clouds, last_clouds)
        change = np.sum(changed_pixels)/pixel_count

        self._last_map = limiting_mag_map

        return float(change)
