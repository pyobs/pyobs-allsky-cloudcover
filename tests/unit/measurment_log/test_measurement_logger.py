import asyncio
import datetime

import pytest

from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy
from pyobs_cloudcover.measurement_log.logger_strategies.memory import MemoryLog
from pyobs_cloudcover.measurement_log.measurment_logger import MeasurementLogger

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


class Flaky(LoggerStrategy):
    """Fails the first ``failures`` writes, like an InfluxDB that is down for a while."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.written: list[datetime.datetime] = []

    def __call__(self, measurements, obs_time) -> None:  # type: ignore[no-untyped-def]
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("influx down")
        self.written.append(obs_time)


async def _drain(logger: MeasurementLogger) -> None:
    for _ in range(200):
        if logger.pending == 0:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_retries_until_the_database_is_back() -> None:
    strategy = Flaky(failures=3)
    logger = MeasurementLogger(strategy, {}, retry_min=0.01, retry_max=0.02)
    task = asyncio.create_task(logger.run())
    try:
        logger.submit({"m": {"f": 1.0}}, T0)
        logger.submit({"m": {"f": 2.0}}, T0 + datetime.timedelta(minutes=1))
        await _drain(logger)
    finally:
        task.cancel()
    assert strategy.written == [T0, T0 + datetime.timedelta(minutes=1)]
    assert logger.failures == 3


@pytest.mark.asyncio
async def test_queue_is_bounded() -> None:
    strategy = Flaky(failures=10**6)
    logger = MeasurementLogger(strategy, {}, max_queue=3, retry_min=0.01)
    for i in range(5):
        logger.submit({}, T0 + datetime.timedelta(minutes=i))
    assert logger.pending == 3


@pytest.mark.asyncio
async def test_submit_does_not_block() -> None:
    class Slow(LoggerStrategy):
        def __call__(self, measurements, obs_time) -> None:  # type: ignore[no-untyped-def]
            import time

            time.sleep(0.3)

    logger = MeasurementLogger(Slow(), {})
    task = asyncio.create_task(logger.run())
    await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    start = loop.time()
    logger.submit({}, T0)
    await asyncio.sleep(0.05)  # the write runs in a thread, so the loop keeps ticking
    assert loop.time() - start < 0.2
    task.cancel()


def test_memory_log() -> None:
    log = MemoryLog()
    log({"m": {"f": 0.5}}, T0)
    assert log.points == [(T0, {"m": {"f": 0.5}})]
