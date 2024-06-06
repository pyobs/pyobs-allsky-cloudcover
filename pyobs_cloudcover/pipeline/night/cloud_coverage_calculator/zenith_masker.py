from copy import copy
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from cloudmap_rs import AltAzCoord


class ZenithMasker(object):
    def __init__(self, altitude: float) -> None:
        self._altitude = altitude

    def __call__(self, image: npt.NDArray[np.float_], alt_az_list: List[List[Optional[AltAzCoord]]]) -> npt.NDArray[np.float_]:
        mask = [[entry.alt < np.deg2rad(self._altitude) if entry is not None else False for entry in row] for row in alt_az_list]

        masked_image = copy(image)
        masked_image[mask] = np.nan

        return masked_image
