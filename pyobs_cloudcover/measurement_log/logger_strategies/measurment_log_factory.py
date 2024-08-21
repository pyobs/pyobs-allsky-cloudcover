from typing import Dict

from pyobs_cloudcover.measurement_log.logger_strategies.influx import Influx
from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy


class MeasurementLogFactory:
    @staticmethod
    def build(config: Dict[str, str]) -> LoggerStrategy:
        if config["type"] == "influx":
            return Influx(
                url=config["url"],
                bucket=config["bucket"],
                org=config["org"],
                token=config["token"]
            )

        raise NotImplementedError(config["type"])