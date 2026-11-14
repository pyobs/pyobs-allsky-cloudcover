"""North-up sky image, timestamps, and display of 8-bit images."""

import datetime
import io

import numpy as np
import pytest
from astroplan import Observer
from PIL import Image

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.render import render_cloud_map, render_frame, render_sky_image
from pyobs_cloudcover.web_api.server import Server
from pyobs_cloudcover.world_model import PointedModel

T = datetime.datetime(2026, 11, 13, 10, 7, 33, tzinfo=datetime.UTC)
W, H = 1152, 648


def _camera() -> PointedModel:
    # the ROTSE mast camera at quarter resolution: image down = az 97, axis ~2 deg off the zenith
    return PointedModel(W / 2, H / 2, 488.0, rotation=np.deg2rad(97), tilt_x=0.025, tilt_y=0.023, k1=0.28, k2=0.2)


def _frame_with_spots(model: PointedModel, spots: list[tuple[float, float, tuple[int, int, int]]]) -> Frame:
    data = np.zeros((H, W, 3))
    for alt, az, colour in spots:
        x, y = model.altaz_to_pix(np.deg2rad(alt), np.deg2rad(az))
        data[int(y) - 6 : int(y) + 7, int(x) - 6 : int(x) + 7] = colour
    return Frame(data, T, saturation=255.0)


def _image(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)


def _centroid(img: np.ndarray, channel: int) -> tuple[float, float]:
    """Centroid of the pure-``channel`` pixels inside the sky circle (the N/E/S/W labels on its edge are red too)."""
    others = [c for c in range(3) if c != channel]
    c = (img.shape[0] - 1) / 2
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    mask = (img[..., channel] > 150) & (img[..., others].max(axis=2) < 80) & (np.hypot(xx - c, yy - c) < 0.9 * c)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0
    return float(xs.mean()), float(ys.mean())


def test_sky_image_is_north_up_east_left() -> None:
    model = _camera()
    # red in the north, green in the east, both at 60 deg altitude
    frame = _frame_with_spots(model, [(60, 0, (255, 0, 0)), (60, 90, (0, 255, 0))])
    size = 480
    img = _image(render_sky_image(frame, model, size=size))
    c = (size - 1) / 2
    r = c * 30 / 90  # 60 deg altitude = 30 deg from the zenith, with the horizon at the edge
    nx, ny = _centroid(img, 0)
    ex, ey = _centroid(img, 1)
    assert (nx, ny) == pytest.approx((c, c - r), abs=4)
    assert (ex, ey) == pytest.approx((c - r, c), abs=4)


def test_sky_image_outside_the_camera_is_black() -> None:
    img = _image(render_sky_image(Frame(np.full((H, W, 3), 200.0), T, saturation=255.0), _camera(), size=480))
    # the horizon (alt ~5 deg) to the south-east is far outside a ~100 x 67 deg field around the zenith
    assert img[400, 100].max() == 0
    # the zenith is inside
    assert img[240, 240].min() > 150


def test_timestamp_on_maps_and_images() -> None:
    empty = SkyPixelQuery(SkyGrid([], []), [])
    with_time = _image(render_cloud_map(empty, time=T))[:16, :150]
    without = _image(render_cloud_map(empty))[:16, :150]
    assert with_time.max() > 100
    assert without.max() == 0
    frame_img = _image(render_frame(Frame(np.full((300, 400, 3), 250.0), T, saturation=255.0)))
    assert frame_img[:17, :150].min() == 0  # dark box behind the time on a bright sky


def test_eight_bit_images_are_shown_unstretched() -> None:
    img = _image(render_frame(Frame(np.full((300, 400, 3), [120.0, 128.0, 136.0]), T, saturation=255.0)))
    assert img[150, 200].tolist() == [120, 128, 136]


@pytest.mark.asyncio
async def test_sky_route(aiohttp_client, observer: Observer) -> None:  # type: ignore[no-untyped-def]
    server = Server(CoverageQueryExecutor(observer), clock=lambda: T + datetime.timedelta(minutes=1))
    client = await aiohttp_client(server._app)
    assert "X-Placeholder" in (await client.get("/sky.png")).headers  # no frame yet
    server.set_frame(Frame(np.full((H, W, 3), 100.0), T, saturation=255.0))
    assert "X-Placeholder" in (await client.get("/sky.png")).headers  # no world model
    server.set_world_model(_camera())
    response = await client.get("/sky.png")
    assert response.status == 200 and response.content_type == "image/png"


def test_every_image_is_labelled_bottom_left() -> None:
    from pyobs_cloudcover.web_api.render import render_placeholder, render_value_map

    empty = SkyPixelQuery(SkyGrid([], []), [])
    frame = Frame(np.full((H, W, 3), 250.0), T, saturation=255.0)
    images = {
        "cloud map": render_cloud_map(empty, size=480),
        "value map": render_value_map(empty, 2, 7, size=480, label="Limiting magnitude", unit="mag"),
        "sky": render_sky_image(frame, _camera(), size=480),
        "placeholder": render_placeholder("x", "y", 480, 480, polar=True, label="Cloud map"),
    }
    for name, png in images.items():
        corner = _image(png)[-30:, :120]
        assert corner.max() > 150, name
    # an unlabelled map has nothing there (the corner is outside the sky circle)
    assert _image(render_cloud_map(empty, size=480, label=""))[-30:, :60].max() < 150
    # on a bright camera image the label sits on a dark box
    corner = _image(render_frame(frame))[-14:, :60]
    assert corner.min() == 0 and ((corner > 150) & (corner < 230)).any()
