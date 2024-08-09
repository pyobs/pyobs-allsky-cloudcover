from unittest.mock import Mock

import pytest
from astroplan import Observer

from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.server import Server


@pytest.mark.asyncio
async def test_point_query_radec(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.point_query_radec = Mock(return_value=False)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/point?ra=0.0&dec=0.0')

    assert response.status == 200
    data = await response.json()

    assert data["value"] is False
    assert data["obs_time"] == 0

@pytest.mark.asyncio
async def test_point_query_altaz(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.point_query_altaz = Mock(return_value=False)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/point?alt=0.0&az=0.0')

    assert response.status == 200
    data = await response.json()

    assert data["value"] is False
    assert data["obs_time"] == 0


@pytest.mark.asyncio
async def test_point_query_invalid(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.point_query_altaz = Mock(return_value=False)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/point?a=0.0&z=0.0')

    assert response.status == 400


@pytest.mark.asyncio
async def test_area_query_radec(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.area_query_radec = Mock(return_value=0.0)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/area?ra=0.0&dec=0.0&radius=10.0')

    assert response.status == 200
    data = await response.json()

    assert data["value"] == 0.0
    assert data["obs_time"] == 0


@pytest.mark.asyncio
async def test_area_query_altaz(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.area_query_altaz = Mock(return_value=0.0)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/area?alt=0.0&az=0.0&radius=10.0')

    assert response.status == 200
    data = await response.json()

    assert data["value"] == 0.0
    assert data["obs_time"] == 0


@pytest.mark.asyncio
async def test_area_query_invalid_coords(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.area_query_altaz = Mock(return_value=0.0)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/area?a=0.0&z=0.0&radius=10.0')

    assert response.status == 400


@pytest.mark.asyncio
async def test_area_query_invalid_coords(aiohttp_client, observer: Observer) -> None:
    executor = CoverageQueryExecutor(observer=observer)
    executor.area_query_altaz = Mock(return_value=0.0)  # type: ignore
    executor.get_obs_time = Mock(return_value=0)

    server = Server(executor)

    client = await aiohttp_client(server._app)
    response = await client.get('/query/area?alt=0.0&az=0.0')

    assert response.status == 400