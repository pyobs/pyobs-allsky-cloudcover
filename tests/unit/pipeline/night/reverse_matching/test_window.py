import numpy as np
import pytest

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow


def test_invalid_image():
    window = ImageWindow(1)

    with pytest.raises(ValueError):
        window(1, 1)


def test_end2end():
    window = ImageWindow(1)
    base_image = np.ones((3, 3))
    padded_image = np.pad(base_image, ((2, 2), (2, 2)), mode='constant', constant_values=0)

    window.set_image(padded_image)

    np.testing.assert_array_equal(window(3, 3), base_image)
