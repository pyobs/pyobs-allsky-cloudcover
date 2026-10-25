import datetime

from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy


class MemoryLog(LoggerStrategy):
    """Keeps all points in memory (for tests and dry runs)."""

    def __init__(self) -> None:
        self.points: list[tuple[datetime.datetime, dict[str, dict[str, float | None]]]] = []

    def __call__(self, measurements: dict[str, dict[str, float | None]], obs_time: datetime.datetime) -> None:
        self.points.append((obs_time, measurements))


class NullLog(LoggerStrategy):
    """Discards everything."""

    def __call__(self, measurements: dict[str, dict[str, float | None]], obs_time: datetime.datetime) -> None:
        pass
