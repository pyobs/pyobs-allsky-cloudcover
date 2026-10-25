from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import (
    SigmaThresholdDetector,
)
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import StarReverseMatcherOptions
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow


class StarReverseMatcherFactory:
    def __init__(self, options: StarReverseMatcherOptions):
        self._options = options

    def __call__(self) -> StarReverseMatcher | ApertureMatcher:
        o = self._options
        if o.type == "aperture":
            return ApertureMatcher(o.snr, o.search_radius, o.aperture_radius, o.annulus, o.max_background)
        detector = SigmaThresholdDetector(o.sigma_threshold, o.distance, o.median_limit)
        max_background = None if o.median_limit is not None else o.max_background
        return StarReverseMatcher(detector, ImageWindow(o.window_size), max_background)
