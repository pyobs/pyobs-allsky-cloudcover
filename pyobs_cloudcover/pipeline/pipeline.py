import abc

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class Pipeline(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frame: Frame) -> CloudCoverageInfo | None: ...
