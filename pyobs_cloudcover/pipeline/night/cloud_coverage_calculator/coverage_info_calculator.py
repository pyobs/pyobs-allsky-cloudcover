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

    def __call__(self, cloud_map: npt.NDArray[np.float_], obs_time: datetime.datetime, alt_az_list: List[List[Optional[AltAzCoord]]]) -> CloudCoverageInfo:
        change = self._coverage_change_calculator(cloud_map)
        coverage = self._coverage_calculator(cloud_map)
        average = float(np.mean(cloud_map, where=~np.isnan(cloud_map)))
        std = float(np.std(cloud_map, where=~np.isnan(cloud_map)))

        zenith = self._zenith_masker(cloud_map, alt_az_list)
        zenith_coverage = self._coverage_calculator(zenith)
        zenith_average = float(np.mean(zenith, where=~np.isnan(zenith)))
        zenith_std = float(np.std(zenith, where=~np.isnan(zenith)))

        return CloudCoverageInfo(copy(cloud_map), coverage, average, std, zenith_coverage, zenith_average, zenith_std, change, obs_time)