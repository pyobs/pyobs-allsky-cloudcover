import numpy as np
from cloudmap_rs import AltAzCoord

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator


def test_call() -> None:
    cloud_map_generator = CloudMapGenerator(20)

    catalog = PixelCatalog(sao=np.array([0, 1]), alt=np.array([85, 85]), az=np.array([0.0, 180]), px=np.array([1, 1]),
                           py=np.array([1, 1]), v_mag=np.array([0, 1]))
    matches = [False, True]

    alt_az_image_list = [
        [None, AltAzCoord(np.deg2rad(89), 3 * np.pi / 2), None],
        [AltAzCoord(np.deg2rad(89), 0), AltAzCoord(np.pi / 2, 0), AltAzCoord(np.deg2rad(89), np.pi)],
        [None, AltAzCoord(np.deg2rad(89), np.pi / 2), None]
    ]

    cloud_map = cloud_map_generator(catalog, matches, alt_az_image_list)

    np.testing.assert_array_almost_equal(cloud_map,
                                         np.array([[np.nan, 0.5, np.nan], [0.6, 0.5, 0.4], [np.nan, 0.5, np.nan]]))


def test_get_integrated_frame() -> None:
    cloud_map_generator = CloudMapGenerator(20, 2)

    sao = np.array([0])
    alt = np.array([0])
    az = np.array([0])
    px = np.array([0])
    py = np.array([0])
    v_mag = np.array([0])

    cloud_map_generator._update_integration_frame(PixelCatalog(sao, alt, az, px, py, v_mag), [True])
    cloud_map_generator._update_integration_frame(PixelCatalog(sao, alt, az, px, py, v_mag), [False])
    cloud_map_generator._update_integration_frame(PixelCatalog.default(), [])

    integrated_catalog, integrated_matches = cloud_map_generator._get_integrated_frame()
    np.testing.assert_array_equal(integrated_catalog.sao, sao)
    np.testing.assert_array_equal(integrated_catalog.alt, alt)
    np.testing.assert_array_equal(integrated_catalog.az, az)
    np.testing.assert_array_equal(integrated_catalog.px, px)
    np.testing.assert_array_equal(integrated_catalog.py, py)
    np.testing.assert_array_equal(integrated_catalog.v_mag, v_mag)

    np.testing.assert_array_equal(integrated_matches, [False])
