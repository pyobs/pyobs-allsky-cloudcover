import abc
import datetime

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class Pipeline(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        ...
