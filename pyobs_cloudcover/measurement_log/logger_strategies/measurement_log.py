import abc
import datetime
from typing import Dict, Optional


class LoggerStrategy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, measurements: Dict[str, Dict[str, Optional[float]]], obs_time: datetime.datetime) -> None:
        ...
