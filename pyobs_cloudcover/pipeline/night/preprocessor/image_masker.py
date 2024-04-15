from copy import copy

import numpy as np
import numpy.typing as npt


class ImageMasker(object):
    def __init__(self, mask: npt.NDArray[np.bool_]) -> None:
        self._mask = mask

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        masked_image = copy(image)
        masked_image *= self._mask

        return masked_image