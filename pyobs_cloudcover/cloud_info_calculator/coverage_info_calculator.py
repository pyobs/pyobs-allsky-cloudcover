import datetime

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator
from pyobs_cloudcover.sky_query import SkyPixelQuery


class CoverageInfoCalculator:
    def __init__(self, coverage_change_calculator: CoverageChangeCalculator) -> None:
        self._coverage_change_calculator = coverage_change_calculator

    def __call__(
        self,
        sky_query: SkyPixelQuery,
        obs_time: datetime.datetime,
        value_query: SkyPixelQuery | None = None,
        value_name: str = "",
        pipeline: str = "",
    ) -> CloudCoverageInfo:
        change = self._coverage_change_calculator(sky_query.get_pixels())

        return CloudCoverageInfo(sky_query, change, obs_time, value_query, value_name, pipeline)
