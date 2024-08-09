import datetime

import numpy as np
from cloudmap_rs import AltAzCoord, SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator.zenith_cloud_coverage_calculator import \
    ZenithCloudCoverageCalculator


class CoverageInfoCalculator:
    def __init__(self, coverage_change_calculator: CoverageChangeCalculator, zenith_coverage: ZenithCloudCoverageCalculator) -> None:
        self._coverage_change_calculator = coverage_change_calculator
        self._zenith_coverage = zenith_coverage

    def __call__(self, sky_query: SkyPixelQuery, obs_time: datetime.datetime) -> CloudCoverageInfo:
        coverage = sky_query.query_radius(AltAzCoord(np.pi/2, 0), np.pi/2)
        change = self._coverage_change_calculator(sky_query.get_pixels())

        zenith_coverage = self._zenith_coverage(sky_query)

        return CloudCoverageInfo(sky_query, coverage, zenith_coverage, change, obs_time)
