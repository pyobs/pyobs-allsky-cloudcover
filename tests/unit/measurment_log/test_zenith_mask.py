import numpy as np

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery


def test_zenith_mask() -> None:
    calculator = ZenithCloudCoverageCalculator(altitude=80)

    alt = np.deg2rad([70, 85, 70, 85, 85, 85, 70, 85, 70])
    pixels = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]

    sky_query = SkyPixelQuery(SkyGrid(alt, np.zeros(9)), pixels)
    coverage_info = CloudCoverageInfo(sky_query, None, None)  # type: ignore[arg-type]
    assert calculator(coverage_info) == 1
