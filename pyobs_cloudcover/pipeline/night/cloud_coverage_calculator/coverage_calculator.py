import numpy as np
import numpy.typing as npt


class CoverageCalculator(object):
    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(self, cloud_map: npt.NDArray[np.float_]) -> float:
        cloud_map_values = cloud_map[~np.isnan(cloud_map)]

        clouds = np.sum((cloud_map_values <= self._threshold).astype(np.uint8))
        free_space = np.sum((cloud_map_values >= 0.0).astype(np.uint8))

        return float(clouds / free_space)
