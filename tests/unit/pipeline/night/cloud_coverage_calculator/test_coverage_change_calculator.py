import numpy as np

from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator


def test_first():
    cloud_map = [True, True, False]
    calculator = CoverageChangeCalculator()
    assert calculator(cloud_map) is None


def test_call_nan_filter():
    calculator = CoverageChangeCalculator()
    calculator._last_map = [True, True, False, False]

    cloud_map = [None, True, True, True]

    assert calculator(cloud_map) == 2/3


def test_multi_call():
    calculator = CoverageChangeCalculator()
    calculator([True, True, False, False])
    assert calculator([False, True, False, True]) == 0.5
    assert calculator([True, True, True, True]) == 0.5
