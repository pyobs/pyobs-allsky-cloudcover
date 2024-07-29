import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


def test_fist():
    cloud_map = [True, False, True, False]
    calculator = CoverageChangeCalculator()
    assert calculator(cloud_map) is None


def test_call():
    calculator = CoverageChangeCalculator()
    calculator._last_map = [True, False, True, False]

    cloud_map = [None, True, True, True]

    assert calculator(cloud_map) == 2/3
