import abc

import numpy as np
import numpy.typing as npt


class WorldModel(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def pix_to_altaz(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        ...

    @abc.abstractmethod
    def altaz_to_pix(self, alt: npt.NDArray[np.float_], az: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        ...
