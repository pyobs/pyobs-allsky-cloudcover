from typing import Dict, Optional

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy
from pyobs_cloudcover.measurement_log.measurement import Measurement


class MeasurementLogger(object):
    def __init__(self, logger: LoggerStrategy, measurements: Dict[str, Measurement]) -> None:
        self._logger = logger
        self._measurements = measurements

    def __call__(self, cloud_info: CloudCoverageInfo) -> None:
        measurements = self._generate_measurements(cloud_info)
        self._logger(measurements, cloud_info.obs_time)

    def _generate_measurements(self, cloud_info: CloudCoverageInfo) -> Dict[str, Dict[str, Optional[float]]]:
        return {
            name: measurement(cloud_info) for name, measurement in self._measurements.items()
        }
