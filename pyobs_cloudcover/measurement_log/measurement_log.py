import abc
import datetime

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class MeasurementLog(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, measurement: CloudCoverageInfo) -> None:
        ...
