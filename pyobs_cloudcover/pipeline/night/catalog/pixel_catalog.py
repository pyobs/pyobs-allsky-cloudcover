from __future__ import annotations

import datetime
from typing import cast

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
        neighbourhood = np.array(
            [np.sum(self._in_neighbourhood(x, y, self.px, self.py, distance)) for x, y in zip(self.px, self.py)
             ]) - 1

        self._filter(neighbourhood == 0)

    @staticmethod
    def _in_neighbourhood(x: float, y: float, px: npt.NDArray[np.float_], py: npt.NDArray[np.float_], d: float) -> npt.NDArray[np.bool_]:
        inside_x_axis: npt.NDArray[np.bool_] = (np.abs(x - px) <= d)
        inside_y_axis: npt.NDArray[np.bool_] = (np.abs(y - py) <= d)
        return inside_x_axis & inside_y_axis

    def filter_window_size(self, height: int, width: int) -> None:
        inside_x_axis = (0 < self.px) & (self.px < width)
        inside_y_axis = (0 < self.py) & (self.py < height)

        self._filter(inside_x_axis & inside_y_axis)

    @classmethod
    def from_altaz_catalog(cls, altaz_catalog: AltAzCatalog, model: WorldModel) -> PixelCatalog:
        px, py = model.altaz_to_pix(np.deg2rad(altaz_catalog.alt), np.deg2rad(altaz_catalog.az))

        return PixelCatalog(
            altaz_catalog.sao,
            cast(npt.NDArray[np.float_], px), cast(npt.NDArray[np.float_], py),
            altaz_catalog.v_mag
            )
