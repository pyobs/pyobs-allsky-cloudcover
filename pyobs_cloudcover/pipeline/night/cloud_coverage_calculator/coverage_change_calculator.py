from typing import Optional

import numpy as np
import numpy.typing as npt


class CoverageChangeCalculator(object):
    def __init__(self) -> None:
        self._last_map: Optional[npt.NDArray[np.float_]] = None

    def __call__(self, cloud_map: npt.NDArray[np.float_]) -> Optional[float]:
        cloud_map_values = cloud_map[~np.isnan(cloud_map)]

        if self._last_map is None:
            self._last_map = cloud_map_values
            return None

        change = np.average(self._last_map - cloud_map_values)
        self._last_map = cloud_map_values

        return float(change)
