import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


def test_fist():
    cloud_map = np.ones((2, 2))
    calculator = CoverageChangeCalculator()
    assert calculator(cloud_map) is None


def test_call():
    calculator = CoverageChangeCalculator()
    calculator._last_map = np.array([2, 2, 2])

    cloud_map = np.ones((2, 2))
    cloud_map[0, 0] = None

    assert calculator(cloud_map) == 1.0
