import abc

import numpy as np
import numpy.typing as npt


class StarDetector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, image: npt.NDArray[np.float_]) -> bool:
        ...
