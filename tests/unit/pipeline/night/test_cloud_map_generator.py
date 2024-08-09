import numpy as np

from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator


def test_cloud_map_generator() -> None:
    generator = CloudMapGenerator(0.5)

    lim_mag_map = np.array([1.0, 0.5, 0.0, np.nan])
    np.testing.assert_array_equal(generator(lim_mag_map), [False, False, True, None])

