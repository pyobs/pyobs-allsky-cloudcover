import numpy as np
import numpy.typing as npt


def calc_color_ratio(image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    return blue/green + blue/red
