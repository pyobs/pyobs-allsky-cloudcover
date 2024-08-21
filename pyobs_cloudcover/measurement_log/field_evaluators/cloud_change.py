from typing import Optional

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class CloudChangeFieldEvaluator(FieldEvaluator):
    def __call__(self, cloud_info: CloudCoverageInfo) -> Optional[float]:
        return cloud_info.change
