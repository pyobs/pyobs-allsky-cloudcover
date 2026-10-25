from typing import cast

import numpy as np

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class TotalCloudCoverageCalculator(FieldEvaluator):
    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        coverage = cloud_info.cloud_cover_query.query_radius(np.pi / 2, 0, np.pi / 2)
        return cast(float | None, coverage)
