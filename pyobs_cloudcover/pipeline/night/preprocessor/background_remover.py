from copy import copy
from typing import Tuple

import numpy as np
import numpy.typing as npt
from astropy.stats import SigmaClip
from photutils.background import MedianBackground, Background2D


class BackgroundRemover(object):
    def __init__(self, sigma_clip: float = 3.0, box_size: Tuple[int, int] = (5, 5)) -> None:
        self._sigma_clip = sigma_clip
        self._box_size = box_size

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        sigma_clip = SigmaClip(sigma=self._sigma_clip)
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, self._box_size, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        corr_image = copy(image)
        corr_image = corr_image - bkg.background

        return corr_image
