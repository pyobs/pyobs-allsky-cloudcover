from typing import Union, Tuple, List, Optional

import numpy as np
from cloudmap_rs import AltAzCoord
from numpy import typing as npt

from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.zenith_masker import ZenithMasker
from pyobs_cloudcover.world_model import WorldModel


class MockWorldModel(WorldModel):
    def __init__(self):
        self._cords = [(1, 1), (1, 2)]

    def pix_to_altaz(self, x: Union[npt.NDArray[np.float_], float], y: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        pass

    def altaz_to_pix(self, alt: Union[npt.NDArray[np.float_], float], az: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:
        return self._cords.pop(0)


def test_zenith_mask() -> None:
    masker = ZenithMasker(altitude=80)
    image = np.ones((3, 3))

    alt_az_list: List[List[Optional[AltAzCoord]]] = [
        [AltAzCoord(70, 0), AltAzCoord(85, 0), AltAzCoord(70, 0)],
        [AltAzCoord(85, 0), AltAzCoord(85, 0), AltAzCoord(85, 0)],
        [AltAzCoord(70, 0), AltAzCoord(85, 0), AltAzCoord(70, 0)]
    ]

    masked_image = masker(image, alt_az_list)
    np.testing.assert_array_equal(masked_image, np.array([[None, 1, None], [1, 1, 1], [None, 1, None]]).astype(np.float_))
