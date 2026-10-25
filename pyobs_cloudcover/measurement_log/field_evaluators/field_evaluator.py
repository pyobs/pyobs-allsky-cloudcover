import abc

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class FieldEvaluator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        ...
