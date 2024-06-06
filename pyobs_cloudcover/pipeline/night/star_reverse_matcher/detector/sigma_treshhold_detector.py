import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.detector import StarDetector


class SigmaThresholdDetector(StarDetector):
    def __init__(self, sigma: float, distance: float, median_limit: float):
        self._sigma = sigma
        self._distance = distance
        self._median_limit = median_limit

    def __call__(self, image: npt.NDArray[np.float_]) -> bool:
        median = np.median(image)
        average = np.average(image)
        std = np.std(image)

        if median > self._median_limit:
            return False

        return True in (image > average + self._sigma * std)
