import asyncio
import datetime
import email.utils
from collections.abc import Callable
from typing import Any

import numpy as np
from aiohttp import web

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.schedule import AnalysisSchedule
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.placeholders import Kind, placeholder_reason
from pyobs_cloudcover.web_api.render import (
    render_cloud_map,
    render_frame,
    render_placeholder,
    render_sky_image,
    render_value_map,
)
from pyobs_cloudcover.world_model.world_model import WorldModel

# side (px) of the sky maps and of the north-up image: all the same, so they can be laid side by side or overlaid
MAP_SIZE = 720
# what each image shows, written into it (bottom left)
LABELS = {
    "image": "Camera image",
    "sky": "Sky (north up)",
    "cloud_map": "Cloud map",
    "limiting_magnitude": "Limiting magnitude",
    "extinction": "Cloud extinction",
}
# limiting magnitudes shown in the PNG map
LIM_MAG_RANGE = (2.0, 7.0)
# cloud extinction (mag) shown in the PNG map
EXTINCTION_RANGE = (0.0, 3.0)


class Server:
    def __init__(
        self,
        query_executor: CoverageQueryExecutor,
        url: str = "localhost",
        port: int = 8080,
        stale_after: float = 900.0,
        clock: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._query_executor = query_executor
        self._stale_after = stale_after
        self._schedule: AnalysisSchedule | None = None
        # current time, for staleness; injectable for tests
        self._clock = clock or (lambda: datetime.datetime.now(datetime.UTC))

        self._url = url
        self._port = port

        self._runner: web.AppRunner | None = None
        self._status: Callable[[], dict[str, Any]] | None = None
        self._images: dict[str, bytes] = {}
        self._frame: Frame | None = None
        self._model: WorldModel | None = None
        self._app = web.Application()
        self._app.add_routes(
            [
                web.get("/query/point", self._point_query),
                web.get("/query/area", self._area_query),
                web.get("/query/limiting_magnitude", self._lim_mag_point),
                web.get("/query/limiting_magnitude/area", self._lim_mag_area),
                web.get("/query/extinction", self._extinction),
                web.get("/status", self._status_route),
                web.get("/weather", self._weather),
                web.get("/cloud_map.png", self._cloud_map_png),
                web.get("/limiting_magnitude.png", self._lim_mag_png),
                web.get("/extinction.png", self._extinction_png),
                web.get("/image.png", self._image_png),
                web.get("/sky.png", self._sky_png),
                web.get("/cloud_map.html", self._cloud_map_html),
            ]
        )

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._query_executor.set_measurement(measurement)
        self._images.clear()

    def set_frame(self, frame: Frame) -> None:
        """The latest analysed camera image, served as ``/image.png`` and (north up) ``/sky.png``."""
        self._frame = frame
        self._images.pop("image", None)
        self._images.pop("sky", None)

    def set_world_model(self, model: WorldModel | None) -> None:
        """The camera's world model (canonical full-resolution pixels), needed for ``/sky.png``."""
        self._model = model
        self._images.pop("sky", None)

    def set_schedule(self, schedule: AnalysisSchedule | None) -> None:
        """When the pipelines run, so placeholders can say which pipeline they are waiting for."""
        self._schedule = schedule

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

    async def _weather(self, request: web.Request) -> web.Response:
        """A flat snapshot for weather aggregators (e.g. pyobs-weather's JSON station): ISO time, cloud cover in
        percent above 30 deg and at the zenith (within 10 deg), and, at night, the zenith sky brightness, mean cloud
        extinction above 30 deg and zenith limiting magnitude."""
        m = self._query_executor.measurement
        if m is None:
            return web.json_response({"time": None, **self._common()}, status=503)
        zenith = np.pi / 2

        def percent(radius_deg: float) -> float | None:
            value = m.cloud_cover_query.query_radius(zenith, 0.0, np.deg2rad(radius_deg))
            return None if value is None else 100.0 * value

        extinction = m.extinction_query.query_radius(zenith, 0.0, np.deg2rad(60)) if m.extinction_query else None
        lim_mag = (
            m.value_query.query_radius(zenith, 0.0, np.deg2rad(10))
            if m.value_query is not None and m.value_name == "limiting_magnitude"
            else None
        )
        return web.json_response(
            {
                "time": m.obs_time.isoformat(),
                "clouds": percent(60),
                "clouds_zenith": percent(10),
                "skymag": m.sky_brightness,
                "extinction": extinction,
                "lim_mag": lim_mag,
                **self._common(),
            }
        )

    async def _status_route(self, request: web.Request) -> web.Response:
        status = self._status() if self._status is not None else {}
        return web.json_response(status)

    # placeholder sizes: maps are square, the camera image keeps the last frame's aspect (16:9 before the first)
    _SIZES = {"cloud_map": MAP_SIZE, "limiting_magnitude": MAP_SIZE, "extinction": MAP_SIZE, "sky": MAP_SIZE}

    def _placeholder(self, kind: Kind) -> web.Response | None:
        status = self._status() if self._status is not None else {}
        error_at = status.get("last_error_at")
        reason = placeholder_reason(
            kind,
            now=self._clock(),
            stale_after=self._stale_after,
            frame_time=self._frame.obs_time if self._frame is not None else None,
            measurement=self._query_executor.measurement,
            last_reason=str(status.get("reason") or ""),
            last_error=str(status.get("last_error") or ""),
            last_error_at=datetime.datetime.fromisoformat(error_at) if error_at else None,
            has_world_model=self._model is not None,
            schedule=self._schedule,
        )
        if reason is None:
            return None
        if kind == "image":
            w, h = (720, 405)
            if self._frame is not None:
                fh, fw = self._frame.data.shape[:2]
                w, h = 720, max(1, round(720 * fh / fw))
        else:
            w = h = self._SIZES[kind]
        body = render_placeholder(
            reason.title, reason.detail, w, h, polar=kind != "image", time=self._clock(), label=LABELS[kind]
        )
        return web.Response(
            body=body, content_type="image/png", headers={"X-Placeholder": reason.title, "Cache-Control": "no-store"}
        )

    def _last_modified(self) -> dict[str, str]:
        """``Last-Modified`` = observation time of the newest camera image this module has (also on placeholders),
        so a dashboard can show how old the camera's latest image is."""
        m = self._query_executor.measurement
        times = [t for t in (self._frame.obs_time if self._frame else None, m.obs_time if m else None) if t]
        if not times:
            return {}
        return {"Last-Modified": email.utils.format_datetime(max(times).astimezone(datetime.UTC), usegmt=True)}

    async def _image(self, kind: Kind) -> web.Response:
        """A rendered image, or a placeholder image saying why there is none (header ``X-Placeholder``). Both carry
        ``Last-Modified`` (see :meth:`_last_modified`)."""
        response = await self._render(kind)
        response.headers.update(self._last_modified())
        return response

    async def _render(self, kind: Kind) -> web.Response:
        placeholder = await asyncio.to_thread(self._placeholder, kind)
        if placeholder is not None:
            return placeholder
        m, frame = self._query_executor.measurement, self._frame
        if kind not in self._images:
            if kind == "image":
                assert frame is not None
                self._images[kind] = await asyncio.to_thread(render_frame, frame)
            elif kind == "sky":
                assert frame is not None and self._model is not None
                self._images[kind] = await asyncio.to_thread(render_sky_image, frame, self._model, MAP_SIZE)
            else:
                assert m is not None
                if kind == "cloud_map":
                    self._images[kind] = await asyncio.to_thread(
                        render_cloud_map, m.cloud_cover_query, MAP_SIZE, time=m.obs_time
                    )
                elif kind == "extinction":
                    assert m.extinction_query is not None
                    self._images[kind] = await asyncio.to_thread(
                        render_value_map,
                        m.extinction_query,
                        *EXTINCTION_RANGE,
                        MAP_SIZE,
                        time=m.obs_time,
                        label=LABELS[kind],
                        unit="mag",
                    )
                else:
                    assert m.value_query is not None
                    self._images[kind] = await asyncio.to_thread(
                        render_value_map,
                        m.value_query,
                        *LIM_MAG_RANGE,
                        MAP_SIZE,
                        time=m.obs_time,
                        label=LABELS[kind],
                        unit="mag",
                    )
        return web.Response(body=self._images[kind], content_type="image/png")

    async def _cloud_map_png(self, _: web.Request) -> web.Response:
        return await self._image("cloud_map")

    async def _lim_mag_png(self, _: web.Request) -> web.Response:
        return await self._image("limiting_magnitude")

    async def _extinction_png(self, _: web.Request) -> web.Response:
        return await self._image("extinction")

    async def _image_png(self, _: web.Request) -> web.Response:
        return await self._image("image")

    async def _sky_png(self, _: web.Request) -> web.Response:
        return await self._image("sky")

    async def _cloud_map_html(self, _: web.Request) -> web.Response:
        page = (
            "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='60'>"
            "<title>Cloud map</title></head><body style='background:#111;color:#ddd;font-family:sans-serif'>"
            "<img src='sky.png' alt='' width='480'> <img src='cloud_map.png' alt='cloud map'> "
            "<img src='limiting_magnitude.png' alt=''> "
            "<img src='extinction.png' alt=''> <img src='image.png' alt='camera image' width='480'></body></html>"
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
