import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog import AltAzCatalog


def test_filter_alt():
    sao = np.array([1, 2])
    alt = np.array([20, 30])
    az = np.array([0, 0])
    v_mag = np.array([0, 0])

    catalog = AltAzCatalog(sao, alt, az, v_mag)
    catalog.filter_alt(25)

    np.testing.assert_array_equal(catalog.sao, sao[1:])
    np.testing.assert_array_equal(catalog.alt, alt[1:])
    np.testing.assert_array_equal(catalog.az, az[1:])
    np.testing.assert_array_equal(catalog.v_mag, v_mag[1:])


def test_filter_v_mag():
    sao = np.array([1, 2])
    alt = np.array([0, 0])
    az = np.array([0, 0])
    v_mag = np.array([6, 5])

    catalog = AltAzCatalog(sao, alt, az, v_mag)
    catalog.filter_v_mag(5.5)

    np.testing.assert_array_equal(catalog.sao, sao[1:])
    np.testing.assert_array_equal(catalog.alt, alt[1:])
    np.testing.assert_array_equal(catalog.az, az[1:])
    np.testing.assert_array_equal(catalog.v_mag, v_mag[1:])
