from copy import copy

import numpy as np
import numpy.typing as npt


class ImageBinner(object):
    def __init__(self, size: int) -> None:
        self._size = size

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        binned_image = copy(image)

        shape = (
            binned_image.shape[0] // self._size, self._size,
            binned_image.shape[1] // self._size, self._size)
        binned_image = binned_image.reshape(shape).mean(-1).mean(1)

        return binned_image
