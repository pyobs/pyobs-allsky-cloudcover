import numpy as np
import numpy.typing as npt


class ImageBinner:
    def __init__(self, size: int) -> None:
        self._size = size

    def __call__(self, image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Mean over size x size blocks. Rows/columns that don't fill a whole block are dropped, which keeps the
        pixel coordinates of everything else unchanged."""
        if self._size <= 1:
            return image

        height, width = image.shape[0] // self._size, image.shape[1] // self._size
        cropped = image[: height * self._size, : width * self._size]
        binned: npt.NDArray[np.float64] = cropped.reshape(height, self._size, width, self._size).mean(axis=(1, 3))

        return binned
