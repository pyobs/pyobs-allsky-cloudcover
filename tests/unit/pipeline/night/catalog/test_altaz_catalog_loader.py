import datetime
from io import StringIO
from unittest.mock import Mock

import numpy as np
from astroplan import Observer
import astropy.units as u

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader


def test_catalog_constructor_from_csv():
    file = StringIO(";SAO;_RAJ2000;_DEJ2000;Vmag\n0;36042;1.0;45.0;6.70\n0;36042;1.0;45.0;6.70")
    catalog = AltAzCatalogLoader.from_csv(file)

    np.testing.assert_array_equal(catalog._sao, [36042, 36042])
    np.testing.assert_array_equal(catalog._ra, [1.0, 1.0])
    np.testing.assert_array_equal(catalog._dec, [45.0, 45.0])
    np.testing.assert_array_equal(catalog._vmag, [6.70, 6.70])


def test_catalog_constructor_call():
    time = datetime.datetime.now()
    observer = Observer(longitude=20.8108 * u.deg, latitude=-32.375823 * u.deg, elevation=1798.0 * u.m, timezone="UTC")

    sao = np.array([1])
    ra = np.array([0])
    dec = np.array([0])
    v_mag = np.array([0])

    constructor = AltAzCatalogLoader(sao, ra, dec, v_mag)
    constructor._equatorial_to_altaz = Mock(return_value=(np.array([0.0]), np.array([0.0])))

    catalog = constructor(observer, time)

    np.testing.assert_array_equal(catalog.sao, sao)
    np.testing.assert_array_equal(catalog.alt, np.array([0.0]))
    np.testing.assert_array_equal(catalog.az, np.array([0.0]))
    np.testing.assert_array_equal(catalog.v_mag, v_mag)
