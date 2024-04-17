import numpy as np

from pyobs_cloudcover.pipeline.night.preprocessor.background_remover import BackgroundRemover


def test_background_remover():
    remover = BackgroundRemover()
    image = np.ones((10, 10))

    corr_image = remover(image)

    np.testing.assert_array_equal(corr_image, np.zeros((10, 10)))
