import abc
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt


class WorldModel(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def pix_to_altaz(self, x: Union[npt.NDArray[np.float_], float], y: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        ...

    @abc.abstractmethod
    def altaz_to_pix(self, alt: Union[npt.NDArray[np.float_], float], az: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        ...
