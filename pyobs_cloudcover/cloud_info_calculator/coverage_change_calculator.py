from typing import Optional

import numpy as np
import numpy.typing as npt


class CoverageChangeCalculator(object):
    def __init__(self) -> None:
        self._last_map: Optional[npt.NDArray[np.float64]] = None

    def __call__(self, new_map: npt.ArrayLike) -> Optional[float]:
        """Fraction of positions known in both maps whose cloud state changed. Maps hold 1.0/0.0/NaN."""
        new = np.array(new_map, dtype=np.float64)

        if self._last_map is None:
            self._last_map = new
            return None

        last = np.asarray(self._last_map, dtype=np.float64)
        image_filter = ~np.isnan(last) & ~np.isnan(new)

        pixel_count = np.sum(image_filter)

        changed_pixels = np.logical_xor(new[image_filter].astype(np.bool_), last[image_filter].astype(np.bool_))
        change = np.sum(changed_pixels)/pixel_count

        self._last_clouds = new

        return float(change)
