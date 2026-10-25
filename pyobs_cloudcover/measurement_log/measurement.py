
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class Measurement:
    def __init__(self, field_evaluators: dict[str, FieldEvaluator]):
        self._field_evaluators = field_evaluators

    def __call__(self, cloud_info: CloudCoverageInfo) -> dict[str, float | None]:
        return {
            name: evaluator(cloud_info) for name, evaluator in self._field_evaluators.items()
        }
