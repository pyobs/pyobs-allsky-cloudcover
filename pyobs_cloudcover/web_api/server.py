from aiohttp import web

from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor


class Server(object):
    def __init__(self, query_executor: CoverageQueryExecutor, url: str = "localhost", port: int = 8080) -> None:
        self._query_executor = query_executor

        self._url = url
        self._port = port

        app = web.Application()
        app.add_routes([web.get("/query", self.query)])
        self._runner = web.AppRunner(app)

    async def start(self) -> None:
        await self._runner.setup()
        site = web.TCPSite(self._runner, 'localhost', self._port)
        await site.start()

    async def query(self, request) -> web.Response:
        ra = float(request.rel_url.query["ra"])
        dec = float(request.rel_url.query["dec"])

        cover = self._query_executor(ra, dec)
        obs_time = self._query_executor.get_obs_time()

        return web.json_response({'value': cover, 'obs_time': obs_time})
