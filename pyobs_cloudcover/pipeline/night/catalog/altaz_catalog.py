import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model import WorldModel


class AltAzCatalog(object):
    def __init__(self, sao: npt.NDArray[np.int_], alt: npt.NDArray[np.float_], az: npt.NDArray[np.float_],
                 v_mag: npt.NDArray[np.float_]) -> None:
        self.sao = sao
        self.alt = alt
        self.az = az
        self.v_mag = v_mag

    def _filter(self, condition: npt.NDArray[np.bool_]) -> None:
        self.sao = self.sao[condition]
        self.alt = self.alt[condition]
        self.az = self.az[condition]
        self.v_mag = self.v_mag[condition]

    def filter_alt(self, alt: float) -> None:
        self._filter(self.alt > alt)

    def filter_v_mag(self, v_mag: float) -> None:
        self._filter(self.v_mag < v_mag)
