import numpy as np

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import \
    SigmaThresholdDetector


def test_call_false():
    detector = SigmaThresholdDetector(3)
    image = np.ones((5, 5))

    assert not detector(image)


def test_call_true():
    detector = SigmaThresholdDetector(3)
    image = np.ones((5, 5))
    image[0][0] = 3

    assert detector(image)
