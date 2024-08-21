from typing import Dict

from astroplan import Observer

from pyobs_cloudcover.measurement_log.field_evaluators.cloud_change import CloudChangeFieldEvaluator
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator
from pyobs_cloudcover.measurement_log.field_evaluators.moon_cloud_coverage_calculator import MoonCloudCoverageCalculator
from pyobs_cloudcover.measurement_log.field_evaluators.sun_cloud_coverage_calculator import SunCloudCoverageCalculator
from pyobs_cloudcover.measurement_log.field_evaluators.total_cloud_coverage_calculator import \
    TotalCloudCoverageCalculator
from pyobs_cloudcover.measurement_log.field_evaluators.zenith_cloud_coverage_calculator import \
    ZenithCloudCoverageCalculator


class FieldEvaluatorFactory:
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, config: Dict[str, str]) -> FieldEvaluator:
        if config["type"] == "total":
            return TotalCloudCoverageCalculator()

        if config["type"] == "zenith":
            return ZenithCloudCoverageCalculator(float(config["altitude"]))

        if config["type"] == "change":
            return CloudChangeFieldEvaluator()

        if config["type"] == "sun":
            return SunCloudCoverageCalculator(self._observer, float(config["radius"]))

        if config["type"] == "moon":
            return MoonCloudCoverageCalculator(self._observer, float(config["radius"]))

        raise ValueError("Invalid Type")
