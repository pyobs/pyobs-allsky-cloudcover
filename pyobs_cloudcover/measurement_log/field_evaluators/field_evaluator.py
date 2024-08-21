import abc
from typing import Optional

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class FieldEvaluator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, cloud_info: CloudCoverageInfo) -> Optional[float]:
        ...
