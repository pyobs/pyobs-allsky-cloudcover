import numpy as np
import numpy.typing as npt


def calc_color_ratio(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    red = image[:, :, 0] + 0.0001
    green = image[:, :, 1] + 0.0001
    blue = image[:, :, 2] + 0.0001

    return blue / green + blue / red
