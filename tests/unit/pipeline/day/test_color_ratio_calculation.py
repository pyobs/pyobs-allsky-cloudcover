import numpy as np

from pyobs_cloudcover.pipeline.day.color_ratio_calculation import calc_color_ratio


def test_color_ratio_calculation():
    image = np.array([[[1, 2, 4]]])
    color_ratio = calc_color_ratio(image)

    np.testing.assert_array_almost_equal(color_ratio, np.array([[6]]), decimal=3)
