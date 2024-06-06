from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import \
    SigmaThresholdDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import StarReverseMatcherOptions
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow


class StarReverseMatcherFactory(object):
    def __init__(self, options: StarReverseMatcherOptions):
        self._options = options

    def __call__(self) -> StarReverseMatcher:
        detector = SigmaThresholdDetector(self._options.sigma_threshold, self._options.distance, self._options.median_limit)
        window = ImageWindow(self._options.window_size)
        reverse_matcher = StarReverseMatcher(detector, window)
        return reverse_matcher
