import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator


def test_call():
    coverage_calculator = CoverageCalculator(0.5)

    cloud_map = np.identity(3) * 0.5
    cloud_map[2, 0] = None
    cloud_map[0, 2] = None

    assert coverage_calculator(cloud_map.astype(np.float_)) == 3/7
