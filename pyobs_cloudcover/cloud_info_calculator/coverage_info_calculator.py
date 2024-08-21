import datetime

import numpy as np
from cloudmap_rs import AltAzCoord, SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.measurement_log.field_evaluators.zenith_cloud_coverage_calculator import \
    ZenithCloudCoverageCalculator


class CoverageInfoCalculator:
    def __init__(self, coverage_change_calculator: CoverageChangeCalculator) -> None:
        self._coverage_change_calculator = coverage_change_calculator

    def __call__(self, sky_query: SkyPixelQuery, obs_time: datetime.datetime) -> CloudCoverageInfo:
        change = self._coverage_change_calculator(sky_query.get_pixels())

        return CloudCoverageInfo(sky_query, change, obs_time)
