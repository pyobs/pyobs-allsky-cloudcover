import datetime
from copy import copy
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from cloudmap_rs import AltAzCoord

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.zenith_masker import ZenithMasker


class CoverageInfoCalculator:
    def __init__(self, coverage_calculator: CoverageCalculator, coverage_change_calculator: CoverageChangeCalculator, zenith_masker: ZenithMasker) -> None:
        self._coverage_calculator = coverage_calculator
        self._coverage_change_calculator = coverage_change_calculator
        self._zenith_masker = zenith_masker

    def __call__(self, limiting_mag_map: npt.NDArray[np.float_], obs_time: datetime.datetime, alt_az_list: List[List[Optional[AltAzCoord]]]) -> CloudCoverageInfo:
        coverage = self._coverage_calculator(limiting_mag_map)
        change = self._coverage_change_calculator(limiting_mag_map)

        zenith = self._zenith_masker(limiting_mag_map, alt_az_list)
        zenith_coverage = self._coverage_calculator(zenith)
        zenith_average = float(np.mean(zenith, where=~np.isnan(zenith)))

        return CloudCoverageInfo(copy(limiting_mag_map), coverage, zenith_coverage, zenith_average, change, obs_time)