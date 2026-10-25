import cv2
import numpy as np
import numpy.typing as npt


def debayer_image(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    debayered_image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    normed_color_image = cv2.normalize(
        debayered_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )  # type: ignore

    return normed_color_image  # type: ignore
