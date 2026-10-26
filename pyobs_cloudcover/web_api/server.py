import asyncio
from collections.abc import Callable
from typing import Any

from aiohttp import web

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.render import render_cloud_map, render_value_map

# limiting magnitudes shown in the PNG map
LIM_MAG_RANGE = (2.0, 7.0)


class Server:
    def __init__(self, query_executor: CoverageQueryExecutor, url: str = "localhost", port: int = 8080) -> None:
        self._query_executor = query_executor

        self._url = url
        self._port = port

        self._runner: web.AppRunner | None = None
        self._status: Callable[[], dict[str, Any]] | None = None
        self._images: dict[str, bytes] = {}
        self._app = web.Application()
        self._app.add_routes(
            [
                web.get("/query/point", self._point_query),
                web.get("/query/area", self._area_query),
                web.get("/query/limiting_magnitude", self._lim_mag_point),
                web.get("/query/limiting_magnitude/area", self._lim_mag_area),
                web.get("/query/extinction", self._extinction),
                web.get("/status", self._status_route),
                web.get("/cloud_map.png", self._cloud_map_png),
                web.get("/limiting_magnitude.png", self._lim_mag_png),
                web.get("/cloud_map.html", self._cloud_map_html),
            ]
        )

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._query_executor.set_measurement(measurement)
        self._images.clear()

    def set_status_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        self._status = provider

    def _common(self) -> dict[str, Any]:
        m = self._query_executor.measurement
        result: dict[str, Any] = {
            "obs_time": self._query_executor.get_obs_time(),
            "pipeline": m.pipeline if m else None,
        }
        if self._status is not None:
            reason = self._status().get("reason")
            if reason:
                result["reason"] = reason
        return result

    def _position(self, request: web.Request) -> tuple[float, float]:
        q = request.rel_url.query

        def get(key: str) -> float | None:
            return float(q[key]) if key in q else None

        return self._query_executor.altaz(get("ra"), get("dec"), get("alt"), get("az"))

    async def _lim_mag_point(self, request: web.Request) -> web.Response:
        try:
            alt, az = self._position(request)
        except ValueError:
            if self._query_executor.measurement is None:
                return web.json_response({"value": None, **self._common()})
            return web.HTTPBadRequest()
        return web.json_response({"value": self._query_executor.limiting_magnitude(alt, az), **self._common()})

    async def _lim_mag_area(self, request: web.Request) -> web.Response:
        if "radius" not in request.rel_url.query:
            return web.HTTPBadRequest()
        radius = float(request.rel_url.query["radius"])
        try:
            alt, az = self._position(request)
        except ValueError:
            if self._query_executor.measurement is None:
                return web.json_response({"mean": None, "min": None, "max": None, **self._common()})
            return web.HTTPBadRequest()
        stats: dict[str, float | None] = {"mean": None, "min": None, "max": None}
        stats.update(self._query_executor.limiting_magnitude_area(alt, az, radius) or {})
        return web.json_response({**stats, **self._common()})

    async def _extinction(self, request: web.Request) -> web.Response:
        q = request.rel_url.query
        radius = float(q["radius"]) if "radius" in q else None
        try:
            alt, az = self._position(request)
        except ValueError:
            if self._query_executor.measurement is None:
                return web.json_response({"value": None, **self._common()})
            return web.HTTPBadRequest()
        m = self._query_executor.measurement
        extra = {"sky_brightness": m.sky_brightness, "zero_point": m.zero_point} if m else {}
        return web.json_response({"value": self._query_executor.extinction(alt, az, radius), **extra, **self._common()})

    async def _status_route(self, request: web.Request) -> web.Response:
        status = self._status() if self._status is not None else {}
        return web.json_response(status)

    async def _png(self, key: str) -> web.Response:
        m = self._query_executor.measurement
        if m is None:
            return web.Response(status=404, text="No measurement yet")
        if key not in self._images:
            if key == "cloud_map":
                self._images[key] = await asyncio.to_thread(render_cloud_map, m.cloud_cover_query)
            else:
                if m.value_query is None or m.value_name != "limiting_magnitude":
                    return web.Response(status=404, text="No limiting-magnitude map (not a night result)")
                self._images[key] = await asyncio.to_thread(render_value_map, m.value_query, *LIM_MAG_RANGE)
        return web.Response(body=self._images[key], content_type="image/png")

    async def _cloud_map_png(self, _: web.Request) -> web.Response:
        return await self._png("cloud_map")

    async def _lim_mag_png(self, _: web.Request) -> web.Response:
        return await self._png("limiting_magnitude")

    async def _cloud_map_html(self, _: web.Request) -> web.Response:
        page = (
            "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='60'>"
            "<title>Cloud map</title></head><body style='background:#111;color:#ddd;font-family:sans-serif'>"
            "<img src='cloud_map.png' alt='cloud map'> <img src='limiting_magnitude.png' alt=''></body></html>"
        )
        return web.Response(text=page, content_type="text/html")

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._url, self._port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

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

        return web.json_response({"value": cover, **self._common(), "obs_time": obs_time})

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

        # 'value' is in percent for compatibility with 1.x clients; 'fraction' is 0..1 like everything else
        return web.json_response(
            {
                "value": None if cover is None else 100.0 * cover,
                "fraction": cover,
                **self._common(),
                "obs_time": obs_time,
            }
        )
