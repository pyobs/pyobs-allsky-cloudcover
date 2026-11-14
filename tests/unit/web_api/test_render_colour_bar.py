"""Value maps (limiting magnitude, extinction) carry a colour bar in the top-right corner."""

import io

import numpy as np
import pytest
from PIL import Image

from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery
from pyobs_cloudcover.web_api.render import RAMP, render_cloud_map, render_value_map, tick_values


def _image(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)


def _empty_query() -> SkyPixelQuery:
    return SkyPixelQuery(SkyGrid([], []), [])


@pytest.mark.parametrize(
    "low, high, expected",
    [
        (2, 7, [2, 3, 4, 5, 6, 7]),
        (0, 3, [0, 1, 2, 3]),
        (0, 10, [0, 2, 4, 6, 8, 10]),
        (0, 0.5, [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        (2.5, 7.3, [3, 4, 5, 6, 7]),
        (0, 100, [0, 20, 40, 60, 80, 100]),
    ],
)
def test_tick_values_are_round(low: float, high: float, expected: list[float]) -> None:
    assert tick_values(low, high) == pytest.approx(expected)


def test_tick_values_degenerate_range() -> None:
    assert tick_values(5, 5) == [5]


@pytest.mark.parametrize("size", [480, 720])
def test_value_map_has_colour_bar_top_right(size: int) -> None:
    image = _image(render_value_map(_empty_query(), 2, 7, size=size, label="Limiting magnitude", unit="mag"))
    length = round(0.2 * size)
    x0, x1 = size - 8 - length, size - 8
    row = image[10, x0:x1]
    # runs from the ramp's low colour (left) to its high colour (right)
    assert np.abs(row[0] - RAMP[0]).max() <= 2
    assert np.abs(row[-1] - RAMP[-1]).max() <= 2
    # tick labels below the bar
    assert image[19:30, x0 - 4 : size].max() > 150
    # everything the bar adds (compared with a map without one) stays outside the sky circle
    plain = _image(render_cloud_map(_empty_query(), size=size, label="Limiting magnitude"))
    ys, xs = np.nonzero((image != plain).any(axis=2))
    c = (size - 1) / 2
    assert len(ys) > 0 and (np.hypot(xs - c, ys - c) > c).all()


def test_cloud_map_has_no_colour_bar() -> None:
    image = _image(render_cloud_map(_empty_query(), size=480))
    assert image[4:30, 380:476].max() < 100
