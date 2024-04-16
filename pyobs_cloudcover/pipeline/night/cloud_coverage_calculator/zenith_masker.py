from copy import copy

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.world_model.world_model import WorldModel


class ZenithMasker(object):
    def __init__(self, altitude: float, model: WorldModel) -> None:
        self._zenith = np.array(model.altaz_to_pix(0.0, 0))
        offset = np.array(model.altaz_to_pix(np.deg2rad(altitude), 0.0))

        self._radius = np.sqrt(np.sum(np.square(self._zenith - offset)))

    def __call__(self, image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        nx, ny = image.shape
        x, y = np.arange(0, nx), np.arange(0, ny)
        x_coordinates, y_coordinates = np.meshgrid(x, y)

        circ_mask = (x_coordinates - self._zenith[0])**2 + (y_coordinates - self._zenith[1])**2 <= self._radius**2
        masked_image = copy(image)
        masked_image[~circ_mask] = None

        return masked_image
