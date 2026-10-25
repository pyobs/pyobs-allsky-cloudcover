import asyncio
import concurrent.futures
import datetime
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from pyobs.events import Event, NewImageEvent
from pyobs.modules import Module

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.measurment_logger_factory import MeasurementLoggerFactory
from pyobs_cloudcover.obs_time import time_from_header
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.web_api.server_factory import ServerFactory

log = logging.getLogger(__name__)


class Application(Module):
    def __init__(self,
                 image_sender: str,
                 server: dict[str, Any],
                 measurement_log: dict[str, Any],
                 pipelines: dict[str, dict[str, Any]], *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        self._image_sender = image_sender

        server_factory = ServerFactory(self.observer)
        self._server = server_factory(server)

        measurement_logger_factory = MeasurementLoggerFactory(self.observer)
        self._measurement_log = measurement_logger_factory(measurement_log)

        pipeline_controller_factory = PipelineControllerFactory(self.observer)
        self._pipeline_controller = pipeline_controller_factory(pipelines)

        # the pipelines keep state between frames, so they run one at a time, off the event loop
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloudcover")
        self._latest: str | None = None
        self._new_image = asyncio.Event()

        self.add_background_task(self._process_images)
        self.add_background_task(self._measurement_log.run)

    async def open(self) -> None:
        await super().open()
        await self._server.start()

        await self.comm.register_event(NewImageEvent, self.process_new_image)

    async def close(self) -> None:
        await self._server.stop()
        await super().close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def process_new_image(self, event: Event, sender: str) -> bool:
        if not isinstance(event, NewImageEvent):
            return False

        if sender != self._image_sender:
            return False

        self.submit_image(event.filename)
        return True

    def submit_image(self, filename: str) -> None:
        """Queue an image for analysis. Only the newest image is kept: if analysis falls behind, older
        images that haven't been started yet are skipped."""
        if self._latest is not None:
            log.info("Skipping %s, a newer image arrived before it was analysed.", self._latest)
        self._latest = filename
        self._new_image.set()

    async def _process_images(self) -> None:
        while True:
            await self._new_image.wait()
            self._new_image.clear()
            filename, self._latest = self._latest, None
            if filename is not None:
                await self.analyse(filename)

    async def analyse(self, filename: str) -> bool:
        """Analyse one image; errors are logged, never raised."""
        try:
            log.info("Analysing %s.", filename)
            image = await self.vfs.read_image(filename)

            obs_time = time_from_header(image.header)
            if obs_time is None:
                log.error("Image %s has no DATE-OBS, skipping.", filename)
                return False

            data = image.data.astype(np.float64)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, self._run_pipeline, data, obs_time)
        except Exception:
            log.exception("Analysing %s failed.", filename)
            return False

        if result is None:
            log.info("No pipeline for the sun altitude of %s.", filename)
            return False

        measurement, fields = result
        self._server.set_measurement(measurement)
        self._measurement_log.submit(fields, measurement.obs_time)
        log.info("Finished %s.", filename)
        return True

    def _run_pipeline(
        self, data: npt.NDArray[np.float64], obs_time: datetime.datetime
    ) -> tuple[CloudCoverageInfo, dict[str, dict[str, float | None]]] | None:
        measurement = self._pipeline_controller(data, obs_time)
        if measurement is None:
            return None
        return measurement, self._measurement_log.evaluate(measurement)
