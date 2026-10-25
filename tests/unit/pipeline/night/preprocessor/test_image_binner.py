import numpy as np

from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner


def test_image_binner():
    binner = ImageBinner(2)
    image = np.identity(4)

    binned_image = binner(image)

    np.testing.assert_array_equal(binned_image, np.identity(2)*0.5)


def test_image_binner_odd_size() -> None:
    # regression: reshape failed unless both sides were multiples of the bin size
    image = np.arange(35, dtype=float).reshape(5, 7)
    binned = ImageBinner(2)(image)
    assert binned.shape == (2, 3)
    assert binned[0, 0] == np.mean([0, 1, 7, 8])
