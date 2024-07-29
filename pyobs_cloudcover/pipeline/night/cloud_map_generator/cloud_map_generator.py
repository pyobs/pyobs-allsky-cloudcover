from typing import List, Optional

import numpy as np
import numpy.typing as npt


class CloudMapGenerator:
    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(self, lim_magnitude_map: npt.NDArray[np.float_]) -> List[Optional[bool]]:
        return [
            bool(value < self._threshold) if not np.isnan(value)
            else None
            for value in lim_magnitude_map
        ]
