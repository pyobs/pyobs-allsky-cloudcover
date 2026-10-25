import numpy as np
import numpy.typing as npt


class CloudMapGenerator:
    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(self, lim_magnitude_map: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """1.0 where the value is below the threshold (cloudy), 0.0 above (clear), NaN where unknown."""
        cloud_map: npt.NDArray[np.float64] = (lim_magnitude_map < self._threshold).astype(np.float64)
        cloud_map[np.isnan(lim_magnitude_map)] = np.nan

        return cloud_map
