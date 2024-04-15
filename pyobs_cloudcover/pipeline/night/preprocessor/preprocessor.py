import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.preprocessor.background_remover import BackgroundRemover
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker


class Preprocessor(object):
    def __init__(self, masker: ImageMasker, binner: ImageBinner, background_remover: BackgroundRemover):
        self._masker = masker
        self._binner = binner
        self._background_remover = background_remover

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        binned_image = self._binner(image)
        masked_image = self._masker(binned_image)
        processed_image = self._background_remover(masked_image)

        return processed_image