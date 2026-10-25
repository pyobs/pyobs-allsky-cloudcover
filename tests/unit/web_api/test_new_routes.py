import datetime
import io

import numpy as np
import pytest
from astroplan import Observer
from PIL import Image

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.sky_query import SkyPixelQuery
from pyobs_cloudcover.status import ModuleStatus
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.server import Server

T = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)


def night_result() -> CloudCoverageInfo:
    grid = SphericalAltAzGenerator(20000, 0)()
    # limiting magnitude 6 in the west (cloud-free), 3 in the east
    lim = np.where(np.rad2deg(grid.az) > 180, 6.0, 3.0)
    cloud = (lim < 4.5).astype(float)
    return CloudCoverageInfo(
        SkyPixelQuery(grid, cloud), None, T, SkyPixelQuery(grid, lim), "limiting_magnitude", "night"
    )


@pytest.fixture
def server(observer: Observer) -> Server:
    s = Server(CoverageQueryExecutor(observer))
    s.set_measurement(night_result())
    return s


@pytest.mark.asyncio
async def test_limiting_magnitude_routes(aiohttp_client, server: Server) -> None:  # type: ignore[no-untyped-def]
    client = await aiohttp_client(server._app)
    west = await (await client.get("/query/limiting_magnitude?alt=45&az=270")).json()
    assert west["value"] == 6.0 and west["pipeline"] == "night" and west["obs_time"] == T.timestamp()
    east = await (await client.get("/query/limiting_magnitude/area?alt=45&az=90&radius=10")).json()
    assert east["mean"] == east["min"] == east["max"] == 3.0
    radec = await (await client.get("/query/limiting_magnitude?ra=100&dec=-30")).json()
    assert radec["value"] in (3.0, 6.0)
    assert (await client.get("/query/limiting_magnitude/area?alt=45&az=90")).status == 400


@pytest.mark.asyncio
async def test_existing_routes_gain_pipeline(aiohttp_client, server: Server) -> None:  # type: ignore[no-untyped-def]
    client = await aiohttp_client(server._app)
    point = await (await client.get("/query/point?alt=45&az=90")).json()
    assert point["value"] is True and point["pipeline"] == "night"


@pytest.mark.asyncio
async def test_png_maps(aiohttp_client, server: Server) -> None:  # type: ignore[no-untyped-def]
    client = await aiohttp_client(server._app)
    for route in ("/cloud_map.png", "/limiting_magnitude.png"):
        response = await client.get(route)
        assert response.status == 200 and response.content_type == "image/png"
        image = Image.open(io.BytesIO(await response.read()))
        rgb = np.asarray(image.convert("RGB")).astype(int)
        h, w, _ = rgb.shape
        # east (left) and west (right) of the zenith differ
        assert np.abs(rgb[h // 2, w // 4] - rgb[h // 2, 3 * w // 4]).sum() > 100
    html = await client.get("/cloud_map.html")
    assert "cloud_map.png" in await html.text()


@pytest.mark.asyncio
async def test_status_and_reason(aiohttp_client, observer: Observer) -> None:  # type: ignore[no-untyped-def]
    server = Server(CoverageQueryExecutor(observer))
    status = ModuleStatus(received=3, without_result=1, last_reason="no pipeline for the sun at -10.0 deg")
    server.set_status_provider(status.to_dict)
    client = await aiohttp_client(server._app)
    data = await (await client.get("/status")).json()
    assert data["frames"]["received"] == 3 and data["reason"].startswith("no pipeline")
    point = await (await client.get("/query/point?alt=90&az=0")).json()
    assert point["value"] is None and "no pipeline" in point["reason"]
    assert (await client.get("/cloud_map.png")).status == 404
