import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


def test_fist():
    cloud_map = np.ones((2, 2))
    calculator = CoverageChangeCalculator(threshold=0.5)
    assert calculator(cloud_map) is None


def test_call():
    calculator = CoverageChangeCalculator(threshold=0.5)
    calculator._last_map = np.identity(2)

    cloud_map = np.ones((2, 2))
    cloud_map[0, 0] = np.nan

    assert calculator(cloud_map) == 2/3
