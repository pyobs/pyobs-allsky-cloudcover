from typing import Any

from pyobs_cloudcover.measurement_log.logger_strategies.influx import Influx
from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy
from pyobs_cloudcover.measurement_log.logger_strategies.memory import MemoryLog, NullLog


class MeasurementLogFactory:
    @staticmethod
    def build(config: dict[str, Any]) -> LoggerStrategy:
        if config["type"] == "influx":
            return Influx(
                url=config["url"],
                bucket=config["bucket"],
                org=config["org"],
                token=config["token"],
                timeout=float(config.get("timeout", 10.0)),
            )
        if config["type"] == "memory":
            return MemoryLog()
        if config["type"] == "none":
            return NullLog()

        raise NotImplementedError(config["type"])
