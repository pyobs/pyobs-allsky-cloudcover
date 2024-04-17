import numpy as np

from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner


def test_image_binner():
    binner = ImageBinner(2)
    image = np.identity(4)

    binned_image = binner(image)

    np.testing.assert_array_equal(binned_image, np.identity(2)*0.5)
