import datetime

import numpy as np
import numpy.typing as npt
from cloudmap_rs import SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


class CoverageInfoCalculator:
    def __init__(self, coverage_change_calculator: CoverageChangeCalculator) -> None:
        self._coverage_change_calculator = coverage_change_calculator

    def __call__(self, sky_query: SkyPixelQuery, cloud_plot: bytes, obs_time: datetime.datetime) -> CloudCoverageInfo:
        change = self._coverage_change_calculator(sky_query.get_pixels())

        return CloudCoverageInfo(sky_query, change, cloud_plot, obs_time)
