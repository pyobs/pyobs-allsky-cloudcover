import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


def test_filter_close() -> None:
    sao = np.array([1, 2, 3])
    alt = np.array([80, 90, 80])
    az = np.array([0, 0, 180])
    px = np.array([10, 10, 20])
    py = np.array([10, 15, 20])
    v_mag = np.array([0, 0, 0])

    catalog = PixelCatalog(sao, alt, az, px, py, v_mag)
    catalog.filter_close(5)

    np.testing.assert_array_equal(catalog.sao, sao[2:])
    np.testing.assert_array_equal(catalog.alt, alt[2:])
    np.testing.assert_array_equal(catalog.az, az[2:])
    np.testing.assert_array_equal(catalog.px, px[2:])
    np.testing.assert_array_equal(catalog.py, py[2:])
    np.testing.assert_array_equal(catalog.v_mag, v_mag[2:])
