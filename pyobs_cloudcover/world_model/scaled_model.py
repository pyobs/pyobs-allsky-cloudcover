from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model.world_model import WorldModel

FloatArray = npt.NDArray[np.float64]


class ScaledModel(WorldModel):
    """Any world model, seen through ``bin_size`` x ``bin_size`` binning.

    Pixel centres are at integer coordinates, so binned pixel ``i`` is centred on original coordinate
    ``i * bin_size + (bin_size - 1) / 2``.
    """

    def __init__(self, model: WorldModel, bin_size: int) -> None:
        if bin_size < 1:
            raise ValueError("bin_size must be >= 1")
        self.model = model
        self.bin_size = bin_size
        self._offset = (bin_size - 1) / 2

    def pix_to_altaz(
        self, x: FloatArray | float, y: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        b = self.bin_size
        return self.model.pix_to_altaz(np.asarray(x) * b + self._offset, np.asarray(y) * b + self._offset)

    def altaz_to_pix(
        self, alt: FloatArray | float, az: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        x, y = self.model.altaz_to_pix(alt, az)
        b = self.bin_size
        return (np.asarray(x) - self._offset) / b, (np.asarray(y) - self._offset) / b
