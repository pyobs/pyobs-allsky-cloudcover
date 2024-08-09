import pytest
from astroplan import Observer
import astropy.units as u


@pytest.fixture(scope='module')
def observer():
    return Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)