import datetime
from copy import copy

import numpy as np
import numpy.typing as npt

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

    def __call__(self, cloud_map: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        change = self._coverage_change_calculator(cloud_map)
        coverage = self._coverage_calculator(cloud_map)
        zenith_coverage = self._coverage_calculator(self._zenith_masker(cloud_map))

        return CloudCoverageInfo(copy(cloud_map), coverage, zenith_coverage, change, obs_time)