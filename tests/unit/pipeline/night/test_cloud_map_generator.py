import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.cloud_map_generator import CloudMapGenerator


def test_call():
    cloud_map_generator = CloudMapGenerator(1)

    catalog = PixelCatalog(sao=np.array([0, 1]), px=np.array([1, 1]), py=np.array([1, 1]), v_mag=np.array([0, 1]))
    matches = [False, True]

    cloud_map = cloud_map_generator(catalog, matches, 3, 3)

    np.testing.assert_array_equal(cloud_map, 0.5 * np.array([[np.nan, 1, np.nan], [1, 1, 1], [np.nan, 1, np.nan]]))
