from aiohttp import web

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor


class Server(object):
    def __init__(self, query_executor: CoverageQueryExecutor, url: str = "localhost", port: int = 8080) -> None:
        self._query_executor = query_executor

        self._url = url
        self._port = port

        self._app = web.Application()
        self._app.add_routes([web.get("/query/point", self._point_query)])
        self._app.add_routes([web.get("/query/area", self._area_query)])

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._query_executor.set_measurement(measurement)

    async def start(self) -> None:
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self._port)
        await site.start()

    async def _point_query(self, request: web.Request) -> web.Response:
        if "ra" in request.rel_url.query and "dec" in request.rel_url.query:
            ra = float(request.rel_url.query["ra"])
            dec = float(request.rel_url.query["dec"])

            cover = self._query_executor.point_query_radec(ra, dec)
        elif "alt" in request.rel_url.query and "az" in request.rel_url.query:
            alt = float(request.rel_url.query["alt"])
            az = float(request.rel_url.query["az"])

            cover = self._query_executor.point_query_altaz(alt, az)
        else:
            return web.HTTPBadRequest()

        obs_time = self._query_executor.get_obs_time()

        return web.json_response({'value': cover, 'obs_time': obs_time})

    async def _area_query(self, request: web.Request) -> web.Response:
        if "radius" in request.rel_url.query:
            radius = float(request.rel_url.query["radius"])
        else:
            return web.HTTPBadRequest()

        if "ra" in request.rel_url.query and "dec" in request.rel_url.query:
            ra = float(request.rel_url.query["ra"])
            dec = float(request.rel_url.query["dec"])

            cover = self._query_executor.area_query_radec(ra, dec, radius)
        elif "alt" in request.rel_url.query and "az" in request.rel_url.query:
            alt = float(request.rel_url.query["alt"])
            az = float(request.rel_url.query["az"])

            cover = self._query_executor.area_query_altaz(alt, az, radius)
        else:
            return web.HTTPBadRequest()

        obs_time = self._query_executor.get_obs_time()

        return web.json_response({'value': cover, 'obs_time': obs_time})