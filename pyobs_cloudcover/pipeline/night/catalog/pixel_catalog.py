from __future__ import annotations

import datetime

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog import AltAzCatalog
from pyobs_cloudcover.pipeline.night.world_model.world_model import WorldModel


class PixelCatalog(object):
    def __init__(self, sao: npt.NDArray[np.int_], px: npt.NDArray[np.float_], py: npt.NDArray[np.float_],
                 v_mag: npt.NDArray[np.float_]) -> None:
        self.sao = sao
        self.px = px
        self.py = py
        self.v_mag = v_mag

    def _filter(self, condition: npt.NDArray[np.bool_]) -> None:
        self.sao = self.sao[condition]
        self.px = self.px[condition]
        self.py = self.py[condition]
        self.v_mag = self.v_mag[condition]

    def filter_close(self, distance: float) -> None:
        def in_neighbourhood(x, y, px, py, d):
            return (np.abs(x - px) <= d) & (np.abs(y - py) <= d)

        neighbourhood = np.array(
            [np.sum(in_neighbourhood(x, y, self.px, self.py, distance)) for x, y in zip(self.px, self.py)
             ]) - 1

        self._filter(neighbourhood == 0)

    @classmethod
    def from_altaz_catalog(cls, altaz_catalog: AltAzCatalog, model: WorldModel) -> PixelCatalog:
        px, py = model.altaz_to_pix(np.deg2rad(altaz_catalog.alt), np.deg2rad(altaz_catalog.az))

        return PixelCatalog(altaz_catalog.sao, px, py, altaz_catalog.v_mag)