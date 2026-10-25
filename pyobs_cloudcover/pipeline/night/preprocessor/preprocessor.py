import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker


class Preprocessor:
    """Turns a frame into the binned luminance image the night pipeline works on.

    Bayer mosaics are binned into (at least) 2x2 superpixels, so ``bin_size`` must be even for them; RGB
    images are averaged over the colour planes first. ``flip_vertical`` and ``masker`` reproduce the 1.x
    preprocessing for legacy configs (vertical flip, then multiplying by a mask in the flipped frame).
    """

    def __init__(self, bin_size: int = 2, flip_vertical: bool = False, masker: ImageMasker | None = None) -> None:
        self._bin_size = bin_size
        self._binner = ImageBinner(bin_size)
        self._flip_vertical = flip_vertical
        self._masker = masker

    @property
    def bin_size(self) -> int:
        return self._bin_size

    def __call__(self, frame: Frame) -> npt.NDArray[np.float64]:
        data = np.asarray(frame.data, dtype=np.float64)
        if data.ndim == 3:
            data = data.mean(axis=2)
        elif frame.bayer is not None and self._bin_size % 2:
            raise ValueError(f"bin_size must be even for Bayer images (got {self._bin_size})")
        if self._flip_vertical:
            data = np.flip(data, axis=0)
        if self._masker is not None:
            data = self._masker(data)
        return self._binner(data)
