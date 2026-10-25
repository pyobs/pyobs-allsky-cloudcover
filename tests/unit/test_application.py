import asyncio
import datetime
import time
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from astropy.io import fits
from pyobs.images import Image

from pyobs_cloudcover.application import Application
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.logger_strategies.memory import MemoryLog
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery

LOCATION = {"longitude": 20.81, "latitude": -32.38, "elevation": 1760.0}


def make_app() -> Application:
    return Application(
        image_sender="camera",
        server={"url": "127.0.0.1", "port": 0},
        measurement_log={
            "logger": {"type": "memory"},
            "measurements": {"cloud": [{"type": "total", "name": "total"}]},
        },
        pipelines={},
        location=LOCATION,
    )


class SlowPipeline:
    """Stands in for the pipeline controller: takes a while, returns a fully cloudy sky."""

    def __init__(self, seconds: float = 0.3, fail: bool = False) -> None:
        self.seconds = seconds
        self.fail = fail
        self.calls: list[datetime.datetime] = []

    def __call__(self, image: Any, obs_time: datetime.datetime) -> CloudCoverageInfo:
        time.sleep(self.seconds)
        if self.fail:
            raise RuntimeError("pipeline exploded")
        self.calls.append(obs_time)
        return CloudCoverageInfo(SkyPixelQuery(SkyGrid([np.pi / 3], [0.0]), [1.0]), None, obs_time)


def fake_image(minute: int) -> Image:
    header = fits.Header({"DATE-OBS": f"2026-03-20T22:{minute:02d}:00"})
    return Image(data=np.zeros((4, 4), dtype=np.uint16), header=header)


@pytest.mark.asyncio
async def test_pipeline_runs_off_the_event_loop() -> None:
    app = make_app()
    app._pipeline_controller = SlowPipeline(0.3)  # type: ignore[assignment]
    app._vfs = AsyncMock()
    app._vfs.read_image = AsyncMock(return_value=fake_image(0))

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    assert await app.analyse("image.fits")
    task.cancel()
    # a blocking pipeline would have frozen the ticker for 0.3 s
    assert ticks > 10


@pytest.mark.asyncio
async def test_errors_are_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    app = make_app()
    app._pipeline_controller = SlowPipeline(0.0, fail=True)  # type: ignore[assignment]
    app._vfs = AsyncMock()
    app._vfs.read_image = AsyncMock(return_value=fake_image(0))
    assert await app.analyse("image.fits") is False
    assert "pipeline exploded" in caplog.text

    app._vfs.read_image = AsyncMock(side_effect=FileNotFoundError("gone"))
    assert await app.analyse("missing.fits") is False


@pytest.mark.asyncio
async def test_only_the_newest_image_is_processed_when_behind() -> None:
    app = make_app()
    pipeline = SlowPipeline(0.2)
    app._pipeline_controller = pipeline  # type: ignore[assignment]
    images = {f"{m}.fits": fake_image(m) for m in range(4)}
    app._vfs = AsyncMock()
    app._vfs.read_image = AsyncMock(side_effect=lambda name: images[name])

    worker = asyncio.create_task(app._process_images())
    app.submit_image("0.fits")
    await asyncio.sleep(0.05)  # 0 is being analysed now
    for name in ("1.fits", "2.fits", "3.fits"):
        app.submit_image(name)
    await asyncio.sleep(0.6)
    worker.cancel()

    assert [t.minute for t in pipeline.calls] == [0, 3]


@pytest.mark.asyncio
async def test_results_go_to_api_and_log() -> None:
    app = make_app()
    app._pipeline_controller = SlowPipeline(0.0)  # type: ignore[assignment]
    app._vfs = AsyncMock()
    app._vfs.read_image = AsyncMock(return_value=fake_image(5))
    writer = asyncio.create_task(app._measurement_log.run())
    assert await app.analyse("image.fits")
    await asyncio.sleep(0.05)
    writer.cancel()

    strategy = app._measurement_log.strategy
    assert isinstance(strategy, MemoryLog)
    obs_time, fields = strategy.points[0]
    assert obs_time == datetime.datetime(2026, 3, 20, 22, 5, tzinfo=datetime.UTC)
    assert fields == {"cloud": {"total": 1.0}}
    assert app._server._query_executor.get_obs_time() == obs_time.timestamp()
