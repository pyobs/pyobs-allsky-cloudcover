from typing import Optional, List

import numpy as np
import numpy.typing as npt


class CoverageChangeCalculator(object):
    def __init__(self) -> None:
        self._last_map: Optional[List[Optional[bool]]] = None

    def __call__(self, new_map: List[Optional[bool]]) -> Optional[float]:
        if self._last_map is None:
            self._last_map = new_map
            return None

        image_filter = np.array([(x is not None) and (y is not None) for x, y in zip(self._last_map, new_map)])

        pixel_count = np.sum(image_filter)

        changed_pixels = np.logical_xor(
            np.array(new_map).astype(np.bool_)[image_filter],
            np.array(self._last_map).astype(np.bool_)[image_filter]
        )
        change = np.sum(changed_pixels)/pixel_count

        self._last_clouds = new_map

        return float(change)
