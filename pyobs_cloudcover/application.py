import logging
from typing import Any

import numpy as np
from pyobs.events import Event, NewImageEvent
from pyobs.modules import Module

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

    async def open(self) -> None:
        await super().open()
        await self._server.start()

        await self.comm.register_event(NewImageEvent, self.process_new_image)

    async def close(self) -> None:
        await self._server.stop()
        await super().close()

    async def process_new_image(self, event: Event, sender: str) -> bool:
        if not isinstance(event, NewImageEvent):
            return False

        if sender != self._image_sender:
            return False

        log.info("Received new image!")

        image = await self.vfs.read_image(event.filename)

        obs_time = time_from_header(image.header)
        if obs_time is None:
            log.error("Image %s has no DATE-OBS, skipping.", event.filename)
            return False

        measurement = self._pipeline_controller(image.data.astype(np.float64), obs_time)

        if measurement is not None:
            self._server.set_measurement(measurement)
            self._measurement_log(measurement)

        log.info("Finished measurement!")
        return True
