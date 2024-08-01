import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


def test_fist():
    cloud_map = np.ones((2, 2))
    calculator = CoverageChangeCalculator(threshold=0.5)
    assert calculator(cloud_map) is None


def test_call_nan_filter():
    calculator = CoverageChangeCalculator(threshold=0.5)
    calculator._last_map = np.identity(2)

    cloud_map = np.ones((2, 2))
    cloud_map[0, 0] = np.nan

    assert calculator(cloud_map) == 2/3

def test_multi_call():
    calculator = CoverageChangeCalculator(threshold=0.5)
    calculator(np.ones((2, 2)))
    assert calculator(np.identity(2)) == 0.5
    assert calculator(np.zeros((2, 2))) == 0.5
