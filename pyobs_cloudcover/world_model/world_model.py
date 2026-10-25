import abc

import numpy as np
import numpy.typing as npt


class WorldModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def pix_to_altaz(self, x: npt.NDArray[np.float64] | float, y: npt.NDArray[np.float64] | float) -> \
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[float, float]:
        ...

    @abc.abstractmethod
    def altaz_to_pix(self, alt: npt.NDArray[np.float64] | float, az: npt.NDArray[np.float64] | float) -> \
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[float, float]:
        ...
