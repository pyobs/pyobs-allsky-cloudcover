from typing import Dict, Optional

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class Measurement:
    def __init__(self, field_evaluators: Dict[str, FieldEvaluator]):
        self._field_evaluators = field_evaluators

    def __call__(self, cloud_info: CloudCoverageInfo) -> Dict[str, Optional[float]]:
        return {
            name: evaluator(cloud_info) for name, evaluator in self._field_evaluators.items()
        }
