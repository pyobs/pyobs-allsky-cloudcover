import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.detector import StarDetector


class SigmaThresholdDetector(StarDetector):
    def __init__(self, sigma: float, distance: float, median_limit: float | None = None):
        self._sigma = sigma
        self._distance = distance
        self._median_limit = median_limit

    def __call__(self, image: npt.NDArray[np.float64], median_limit: float | None = None) -> bool:
        median = np.median(image)

        limit = self._median_limit if median_limit is None else median_limit
        if limit is not None and median > limit:
            return False

        average = np.average(image)
        std = np.std(image)

        nx, ny = image.shape
        center_x, center_y = (nx//2, ny//2)
        x, y = np.arange(0, nx), np.arange(0, ny)
        x_grid, y_grid = np.meshgrid(x, y)
        circ_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 < self._distance**2

        return True in (circ_mask.transpose() * image > average + self._sigma * std)
