import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.detector import StarDetector


class SigmaThresholdDetector(StarDetector):
    def __init__(self, sigma: float):
        self._sigma = sigma

    def __call__(self, image: npt.NDArray[np.float_]) -> bool:
        average = np.average(image)
        std = np.std(image)

        return True in (image > average + self._sigma * std)
