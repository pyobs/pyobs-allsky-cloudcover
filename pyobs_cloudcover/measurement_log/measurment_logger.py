import asyncio
import collections
import datetime
import logging

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy
from pyobs_cloudcover.measurement_log.measurement import Measurement

log = logging.getLogger(__name__)

Fields = dict[str, dict[str, float | None]]


class MeasurementLogger:
    """Evaluates the configured measurements and writes them through a logger strategy (e.g. InfluxDB).

    Writes never block the caller and never raise: points are queued and written by :meth:`run` (a background
    task) in a worker thread. A failed write is retried with exponential backoff (``retry_min`` doubling up to
    ``retry_max`` seconds) while newer points keep queueing; if more than ``max_queue`` points pile up, the oldest
    are dropped.
    """

    def __init__(
        self,
        logger: LoggerStrategy,
        measurements: dict[str, Measurement],
        max_queue: int = 10000,
        retry_min: float = 2.0,
        retry_max: float = 300.0,
    ) -> None:
        self._logger = logger
        self._measurements = measurements
        self._queue: collections.deque[tuple[Fields, datetime.datetime]] = collections.deque()
        self._max_queue = max_queue
        self._retry_min = retry_min
        self._retry_max = retry_max
        self._wakeup: asyncio.Event | None = None
        self.failures = 0

    @property
    def strategy(self) -> LoggerStrategy:
        return self._logger

    @property
    def pending(self) -> int:
        return len(self._queue)

    def evaluate(self, cloud_info: CloudCoverageInfo) -> Fields:
        return {name: measurement(cloud_info) for name, measurement in self._measurements.items()}

    def __call__(self, cloud_info: CloudCoverageInfo) -> None:
        self.submit(self.evaluate(cloud_info), cloud_info.obs_time)

    def submit(self, fields: Fields, obs_time: datetime.datetime) -> None:
        if len(self._queue) >= self._max_queue:
            self._queue.popleft()
            log.warning("Measurement queue full (%d), dropped the oldest point.", self._max_queue)
        self._queue.append((fields, obs_time))
        if self._wakeup is not None:
            self._wakeup.set()

    async def run(self) -> None:
        """Write queued points forever; meant to run as a background task."""
        self._wakeup = asyncio.Event()
        delay = self._retry_min
        while True:
            if not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            fields, obs_time = self._queue[0]
            try:
                await asyncio.to_thread(self._logger, fields, obs_time)
            except Exception as e:
                self.failures += 1
                log.warning(
                    "Writing measurements failed (%s: %s); %d point(s) queued, retrying in %.0f s.",
                    type(e).__name__, e, len(self._queue), delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_max)
                continue
            if delay > self._retry_min:
                log.info("Writing measurements works again.")
            delay = self._retry_min
            # the point may have been dropped meanwhile if the queue overflowed
            if self._queue and self._queue[0][1] is obs_time:
                self._queue.popleft()
