import abc
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt


class WorldModel(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def pix_to_altaz(self, x: Union[npt.NDArray[np.float64], float], y: Union[npt.NDArray[np.float64], float]) -> \
            Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], Tuple[float, float]]:
        ...

    @abc.abstractmethod
    def altaz_to_pix(self, alt: Union[npt.NDArray[np.float64], float], az: Union[npt.NDArray[np.float64], float]) -> \
            Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], Tuple[float, float]]:
        ...
