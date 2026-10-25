import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import (
    LimMagnitudeMapGenerator,
)
from pyobs_cloudcover.sky_query import SkyGrid


def test_call() -> None:
    cloud_map_generator = LimMagnitudeMapGenerator(20)

    catalog = PixelCatalog(sao=np.array([0, 1]), alt=np.array([85, 85]), az=np.array([0.0, 180]), px=np.array([1, 1]),
                           py=np.array([1, 1]), v_mag=np.array([0, 1]))
    matches = [False, True]

    grid = SkyGrid(
        np.deg2rad([89, 89, 90, 89, 89]),
        [3 * np.pi / 2, 0, 0, np.pi, np.pi / 2],
    )

    cloud_map = cloud_map_generator(catalog, matches, grid)

    # nearer stars weigh more: exp(-d/20deg) for the 4 deg (mag 0) and 6 deg (mag 1) stars
    near, far = np.exp(-4 / 20), np.exp(-6 / 20)
    expected = far / (near + far)
    np.testing.assert_array_almost_equal(cloud_map, np.array([0.5, expected, 0.5, 1 - expected, 0.5]))


def test_get_integrated_frame() -> None:
    cloud_map_generator = LimMagnitudeMapGenerator(20, 2)

    sao = np.array([0, 1])
    alt = np.array([0] * 2)
    az = np.array([0] * 2)
    px = np.array([0] * 2)
    py = np.array([0] * 2)
    v_mag = np.array([0] * 2)

    cloud_map_generator._update_integration_frame(PixelCatalog.default(), [])
    cloud_map_generator._update_integration_frame(PixelCatalog(sao, alt, az, px, py, v_mag), [True, True])
    cloud_map_generator._update_integration_frame(PixelCatalog(sao, alt, az, px, py, v_mag), [False, True])


    integrated_matches = cloud_map_generator._get_integrated_frame()

    np.testing.assert_array_equal(integrated_matches, [False, True])
