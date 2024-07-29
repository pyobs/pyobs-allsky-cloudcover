import datetime
from unittest.mock import Mock

import astropy.units as u
import numpy as np
import pytest
from astroplan import Observer
from cloudmap_rs import AltAzCoord, SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor


@pytest.fixture()
def observer():
    return Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)


@pytest.fixture()
def obs_time():
    return datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)



def test_call_no_map(observer, obs_time):
    executor = CoverageQueryExecutor(observer)

    assert executor(10, 10) is None


def test_call(observer, obs_time) -> None:
    executor = CoverageQueryExecutor(observer)
    executor._radec_to_altaz = Mock(return_value=(np.pi/2, 0))  # type: ignore

    cloud_query = SkyPixelQuery([AltAzCoord(0, 0)], [True])
    executor.set_measurement(CloudCoverageInfo(cloud_query,  1, 2, 0.1, obs_time))

    assert executor(10, 10) is True


def test_call_nan(observer, obs_time) -> None:
    executor = CoverageQueryExecutor(observer)
    executor._radec_to_altaz = Mock(return_value=(np.pi/2, 0))  # type: ignore

    cloud_query = SkyPixelQuery([AltAzCoord(0, 0)], [None])
    executor.set_measurement(CloudCoverageInfo(cloud_query,  1, 2, 0.1, obs_time))

    assert executor(10, 10) is None
