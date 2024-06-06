import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator


def test_call() -> None:
    coverage_calculator = CoverageCalculator(0.5)

    cloud_map = np.identity(3)
    cloud_map[2, 0] = None
    cloud_map[0, 2] = None

    np.testing.assert_almost_equal(coverage_calculator(cloud_map.astype(np.float_)), 4/7)
