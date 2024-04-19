from typing import List

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.detector import StarDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow


class StarReverseMatcher(object):
    def __init__(self, detector: StarDetector, window: ImageWindow):
        self._detector = detector
        self._window = window

    def __call__(self, image: npt.NDArray[np.float_], catalog: PixelCatalog) -> List[bool]:
        self._window.set_image(image)

        found = [
            self._detector(self._window(px, py)) for px, py in zip(catalog.px, catalog.py)
        ]

        return found
