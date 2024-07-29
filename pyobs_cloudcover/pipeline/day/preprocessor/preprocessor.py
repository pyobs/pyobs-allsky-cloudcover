import cv2
import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker


class Preprocessor:
    def __init__(self, masker: ImageMasker):
        self._masker = masker

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        debayered_image = self._debayer(image)
        masked_image = self._masker(debayered_image)

        return masked_image

    @staticmethod
    def _debayer(image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        debayered_image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        normed_color_image = cv2.normalize(debayered_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return normed_color_image
