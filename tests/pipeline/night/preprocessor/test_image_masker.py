import numpy as np

from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker


def test_image_masker():
    mask = np.array([[True, False], [False, True]])
    masker = ImageMasker(mask)

    image = np.ones((2, 2))
    masked_image = masker(image)

    np.testing.assert_array_equal(masked_image, np.array([[1, 0], [0, 1]]))
