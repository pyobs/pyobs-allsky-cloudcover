from typing import Dict, Any

from astroplan import Observer

from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluators_factory import FieldEvaluatorFactory
from pyobs_cloudcover.measurement_log.logger_strategies.measurment_log_factory import MeasurementLogFactory
from pyobs_cloudcover.measurement_log.measurement import Measurement
from pyobs_cloudcover.measurement_log.measurment_logger import MeasurementLogger


class MeasurementLoggerFactory:
    def __init__(self, observer: Observer) -> None:
        self._field_evaluator_factory = FieldEvaluatorFactory(observer)

    def __call__(self, config: Dict[str, Dict[str, Any]]) -> MeasurementLogger:
        logger = MeasurementLogFactory.build(config["logger"])

        measurements = {
           name: Measurement(
               {
                field_config["name"]: self._field_evaluator_factory(field_config) for field_config in measurement
            }
           ) for name, measurement in config["measurements"].items()
        }

        return MeasurementLogger(logger, measurements)
