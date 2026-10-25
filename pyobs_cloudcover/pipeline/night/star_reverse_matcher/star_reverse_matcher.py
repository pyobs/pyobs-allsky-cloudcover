import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.detector import StarDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow


class StarReverseMatcher:
    """Tests a window around each predicted star position with a detector.

    ``max_background`` (fraction of the saturation level) skips windows whose median is too bright, e.g. near
    the moon; it applies unless the detector has an absolute ``median_limit`` (1.x configs).
    """

    def __init__(self, detector: StarDetector, window: ImageWindow, max_background: float | None = None):
        self._detector = detector
        self._window = window
        self._max_background = max_background

    def __call__(
        self, image: npt.NDArray[np.float64], catalog: PixelCatalog, saturation: float | None = None
    ) -> list[bool]:
        self._window.set_image(image)
        limit = None
        if self._max_background is not None and saturation is not None:
            limit = self._max_background * saturation

        return [self._detector(self._window(px, py), limit) for px, py in zip(catalog.px, catalog.py)]
