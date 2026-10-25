from typing import cast

import numpy as np

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class ZenithCloudCoverageCalculator(FieldEvaluator):
    def __init__(self, altitude: float) -> None:
        self._altitude = altitude

    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        radius = np.pi/2 - np.deg2rad(self._altitude)
        zenith_cover = cloud_info.cloud_cover_query.query_radius(np.pi/2, 0, radius)
        return cast(float | None, zenith_cover)
