from typing import List, Optional

import numpy as np
import numpy.typing as npt


class CloudMapGenerator:
    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(self, lim_magnitude_map: npt.NDArray[np.float_]) -> npt.NDArray[np.bool_]:
        np_cloud_map: npt.NDArray[np.bool_] = lim_magnitude_map < self._threshold
        cloud_map = np_cloud_map.astype(dtype=object)   # Necessary to get the python bool type and allow for None vals

        cloud_map[np.isnan(lim_magnitude_map)] = None

        return cloud_map
