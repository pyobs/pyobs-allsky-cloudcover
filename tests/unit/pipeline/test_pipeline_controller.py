import datetime

import astropy.units as u
import numpy as np
import pytest
from astroplan import Observer
from astropy.coordinates import SkyCoord
from numpy import typing as npt

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.pipeline.pipeline_controller import PipelineController


class MockPipeline(Pipeline):

    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        time = datetime.datetime(2024, 1, 1, 0, 0, 0)
        return CloudCoverageInfo(np.array([]), 0.1, time)


@pytest.fixture()
def observer() -> Observer:
    return Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)


def test_invalid_init_args_list(observer) -> None:
    pipelines = [MockPipeline()]
    interval = []

    with pytest.raises(ValueError):
        PipelineController(pipelines, interval, observer)


def test_invalid_init_overlapping_intervals(observer) -> None:
    pipelines = [MockPipeline(), MockPipeline()]
    interval = [Interval(None, 10), Interval(0, None)]

    with pytest.raises(ValueError):
        PipelineController(pipelines, interval, observer)


def test_pipeline_call_outside_interval(mocker, observer) -> None:
    mocker.patch.object(observer, "sun_altaz", return_value=SkyCoord(alt=10, az=0, frame="altaz", unit="deg"))

    pipelines = [MockPipeline()]
    interval = [Interval(0, 1)]
    controller = PipelineController(pipelines, interval, observer)

    assert controller(np.array([]), datetime.datetime.now()) is None


def test_pipeline_call_inside_interval(mocker, observer) -> None:
    mocker.patch.object(observer, "sun_altaz", return_value=SkyCoord(alt=5, az=0, frame="altaz", unit="deg"))

    pipelines = [MockPipeline()]
    interval = [Interval(0, 10)]
    controller = PipelineController(pipelines, interval, observer)

    assert isinstance(controller(np.array([]), datetime.datetime.now()), CloudCoverageInfo)
