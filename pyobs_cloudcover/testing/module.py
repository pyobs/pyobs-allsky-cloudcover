import asyncio
import logging
from typing import Any

import aiohttp
import numpy as np
from pyobs.events import NewImageEvent
from pyobs.images import Image
from pyobs.modules import Module
from testcontainers.influxdb2 import InfluxDb2Container

import pyobs_cloudcover.application

log = logging.getLogger(__name__)


class TestModule(Module):
    _CACHE_PATH = "/cache/new_image.fits"
    _INFLUX_TOKEN = "TOKEN"
    _INFLUX_ORG = "IAG"
    _INFLUX_BUCKET = "allsky"

    def __init__(self, image_path: str, total_fraction: float, zenith_fraction: float, zenith_value: bool, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._image_path = image_path

        self._total_fraction = total_fraction
        self._zenith_fraction = zenith_fraction
        self._zenith_value = zenith_value

        #self.add_background_task(self.start, restart=False, autostart=True)

    async def open(self) -> None:
        await Module.open(self)

        await self.start()

    async def start(self) -> None:
        with InfluxDb2Container(
            init_mode="setup",
            admin_token=self._INFLUX_TOKEN,
            org_name=self._INFLUX_ORG,
            bucket=self._INFLUX_BUCKET,
            username="test",
            password="5_B2D_Y-Y7hL"  # "Valid" pwd needed here, otherwise the Container won't start
        ) as influx:

            self._override_measurement_log(influx)

            await self._submit_image()

            log.info("Waiting for processing...")
            await asyncio.sleep(10)

            self._check_influx_entries(influx)

        await self._check_http_server()

    def _override_measurement_log(self, influx: InfluxDb2Container) -> None:
        """
        This exploits the memory locality of other modules when using the LocalComm as a comm object.
        Since this test uses test containers, the url of the InfluxDB container is only known at run time.
        Therefore, the InfluxDB parameters are updated here manually.
        """
        cloud_cover: pyobs_cloudcover.application.Application = self.comm._network._clients["cloudcover"].module
        cloud_cover._measurement_log._client, _ = influx.get_client(token=self._INFLUX_TOKEN)
        cloud_cover._measurement_log._bucket = self._INFLUX_BUCKET
        cloud_cover._measurement_log._org = self._INFLUX_ORG

    async def _submit_image(self) -> None:
        image = Image.from_file(self._image_path)
        await self.vfs.write_image(self._CACHE_PATH, image)
        await self.comm.send_event(NewImageEvent(self._CACHE_PATH))

    def _check_influx_entries(self, influx: InfluxDb2Container) -> None:
        client, organization = influx.get_client(token=self._INFLUX_TOKEN, org_name=self._INFLUX_ORG)
        query = ('from(bucket:"allsky")\
                |> range(start: -1y)')

        query_api = client.query_api()
        result = query_api.query(query, org=self._INFLUX_ORG)

        total_cover_table = result[0].records
        zenith_cover_table = result[1].records

        np.testing.assert_almost_equal(total_cover_table[0].get_value(), self._total_fraction, 1)
        np.testing.assert_almost_equal(zenith_cover_table[0].get_value(), self._zenith_fraction, 1)

        log.info("INFLUX TEST SUCCESS!")

    async def _check_http_server(self) -> None:
        cloud_cover: pyobs_cloudcover.application.Application = self.comm._network._clients["cloudcover"].module
        url = cloud_cover._server._url
        port = cloud_cover._server._port

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{url}:{port}/query?ra=0.0&dec=90.0") as resp:
                data = await resp.json()

        assert data["value"] == self._zenith_value

        log.info("SERVER TEST SUCCESS!")
