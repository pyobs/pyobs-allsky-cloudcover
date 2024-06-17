import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


def test_filter_close() -> None:
    sao = np.array([1, 2, 3])
    alt = np.array([80, 90, 80])
    az = np.array([0, 0, 180])
    px = np.array([10, 10, 20])
    py = np.array([10, 15, 20])
    v_mag = np.array([1, 0, 0])

    catalog = PixelCatalog(sao, alt, az, px, py, v_mag)
    catalog.filter_close(5)

    np.testing.assert_array_equal(catalog.sao, sao[1:])
    np.testing.assert_array_equal(catalog.alt, alt[1:])
    np.testing.assert_array_equal(catalog.az, az[1:])
    np.testing.assert_array_equal(catalog.px, px[1:])
    np.testing.assert_array_equal(catalog.py, py[1:])
    np.testing.assert_array_equal(catalog.v_mag, v_mag[1:])


def test_find_cluster() -> None:
    neighbours = [(1, 2), (6, 7), (2, 4, 5), (1, 2)]

    clusters = PixelCatalog._find_cluster(neighbours)  # type: ignore

    assert set(map(lambda x: tuple(x), clusters)) == {(1, 2, 4, 5), (6, 7)}
