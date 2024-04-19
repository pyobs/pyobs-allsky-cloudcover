import datetime
from typing import Dict, Any

from pyobs.events import NewImageEvent, Event
from pyobs.modules import Module
from pyobs.object import get_object

from pyobs_cloudcover.measurement_log.influx import Influx
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.web_api.server import Server
from pyobs_cloudcover.web_api.server_factory import ServerFactory
from pyobs_cloudcover.world_model import WorldModel


class Application(Module):
    def __init__(self,
                 image_sender: str,
                 model: Dict[str, Any],
                 server: Dict[str, Any],
                 measurement_log: Dict[str, Any],
                 pipelines: Dict[str, Dict[str, Any]]) -> None:

        super(Application, self).__init__()

        self._image_sender = image_sender

        world_model: WorldModel = get_object(model, WorldModel)

        server_factory = ServerFactory(self.observer, world_model)
        self._server = server_factory(server)

        self._measurement_log = Influx(**measurement_log)

        pipeline_controller_factory = PipelineControllerFactory(self.observer, world_model)
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

        image = await self.vfs.read_image(event.filename)

        obs_time = datetime.datetime.strptime(image.header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f")

        measurement = self._pipeline_controller(image.data, obs_time)

        if measurement is not None:
            self._server.set_measurement(measurement)