import abc
import datetime


class LoggerStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, measurements: dict[str, dict[str, float | None]], obs_time: datetime.datetime) -> None:
        ...
