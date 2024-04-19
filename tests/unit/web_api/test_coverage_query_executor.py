import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from astroplan import Observer

import astropy.units as u

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.world_model.simple_model import SimpleModel
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor


@pytest.fixture()
def observer():
    return Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)


@pytest.fixture()
def obs_time():
    return datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)


@pytest.fixture()
def model():
    model_parameters = [4.81426598e-03, 2.00000000e+00, 1.06352627e+03, 7.57115607e+02, 5.11194838e+02]
    return SimpleModel(*model_parameters)


def test_call_no_map(observer, obs_time, model):
    window = ImageWindow(2)

    executor = CoverageQueryExecutor(model, observer, window)

    assert executor(10, 10) is None


def test_call_map(observer, obs_time, model):
    window = ImageWindow(1)

    executor = CoverageQueryExecutor(model, observer, window)
    executor._radec_to_altaz = Mock(return_value=(np.pi/2, 0))
    executor._model.altaz_to_pix = Mock(return_value=(1, 1))

    executor.set_measurement(CloudCoverageInfo(np.identity(3), 0, 0, 0, obs_time))

    assert executor(10, 10) == 1/3
    executor._radec_to_altaz.assert_called_once_with(10, 10, obs_time)
    executor._model.altaz_to_pix.assert_called_once_with(np.pi/2, 0)


def test_call_map_out_of_bounds(observer, obs_time, model):
    window = ImageWindow(1)

    executor = CoverageQueryExecutor(model, observer, window)
    executor._radec_to_altaz = Mock(return_value=(np.pi/2, 0))
    executor._model.altaz_to_pix = Mock(return_value=(5, 5))

    executor.set_measurement(CloudCoverageInfo(np.identity(3), 0, 0, 0, obs_time))

    assert executor(10, 10) is None


def test_call_map_nan(observer, obs_time, model):
    window = ImageWindow(1)

    executor = CoverageQueryExecutor(model, observer, window)
    executor._radec_to_altaz = Mock(return_value=(np.pi/2, 0))
    executor._model.altaz_to_pix = Mock(return_value=(1, 1))

    cloud_map = np.identity(3)
    cloud_map[0, 0] = np.nan
    executor.set_measurement(CloudCoverageInfo(cloud_map, 0, 0, 0, obs_time))

    assert executor(10, 10) == 2/8
