from typing import Optional

import numpy as np
import numpy.typing as npt


class ImageWindow(object):
    def __init__(self, h_size: float) -> None:
        self._h_size = h_size
        self._image: Optional[npt.NDArray[np.float_]] = None

    def get_size(self) -> float:
        return self._h_size

    def set_image(self, image: npt.NDArray[np.float_]) -> None:
        self._image = image

    def __call__(self, px: float, py: float) -> npt.NDArray[np.float_]:
        if self._image is None:
            raise ValueError("Image must be set before calling this method.")

        low_px = abs(int(px - self._h_size))
        high_px = int(px + self._h_size)
        low_py = abs(int(py - self._h_size))
        high_py = int(py + self._h_size)

        return self._image[low_py:high_py+1, low_px:high_px+1]
