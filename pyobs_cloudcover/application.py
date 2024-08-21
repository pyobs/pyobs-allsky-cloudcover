import datetime
import logging
from typing import Dict, Any

from pyobs.events import NewImageEvent, Event
from pyobs.modules import Module

from pyobs_cloudcover.measurement_log.logger_strategies.influx import Influx
from pyobs_cloudcover.measurement_log.measurment_logger_factory import MeasurementLoggerFactory
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.web_api.server_factory import ServerFactory

log = logging.getLogger(__name__)


class Application(Module):
    def __init__(self,
                 image_sender: str,
                 server: Dict[str, Any],
                 measurement_log: Dict[str, Any],
                 pipelines: Dict[str, Dict[str, Any]], *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        self._image_sender = image_sender

        server_factory = ServerFactory(self.observer)
        self._server = server_factory(server)

        measurement_logger_factory = MeasurementLoggerFactory(self.observer)
        self._measurement_log = measurement_logger_factory(measurement_log)

        pipeline_controller_factory = PipelineControllerFactory(self.observer)
        self._pipeline_controller = pipeline_controller_factory(pipelines)

    async def open(self) -> None:
        await super(Application, self).open()
        await self._server.start()

        await self.comm.register_event(NewImageEvent, self.process_new_image)

    async def process_new_image(self, event: Event, sender: str) -> None:
        if not isinstance(event, NewImageEvent):
            return

        if sender != self._image_sender:
            return

        log.info("Received new image!")

        image = await self.vfs.read_image(event.filename)

        obs_time = datetime.datetime.strptime(image.header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f")

        measurement = self._pipeline_controller(image.data, obs_time)

        if measurement is not None:
            self._server.set_measurement(measurement)
            self._measurement_log(measurement)

        log.info("Finished measurement!")
