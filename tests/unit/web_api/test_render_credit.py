"""Every rendered image carries the package name and version in its bottom-right corner."""

import datetime
import io
from importlib import metadata

import numpy as np
from PIL import Image

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery
from pyobs_cloudcover.web_api.render import credit_lines, render_cloud_map, render_frame, render_value_map


def _corner(png: bytes) -> np.ndarray:
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    return image[-26:, -150:]


def _empty_query() -> SkyPixelQuery:
    return SkyPixelQuery(SkyGrid([], []), [])


def test_credit_names_package_and_version() -> None:
    name, version = credit_lines()
    assert name == "pyobs-allsky-cloudcover"
    assert version == "v" + metadata.version("pyobs-allsky-cloudcover")


def test_maps_carry_the_credit() -> None:
    # an empty map is black outside the rings, so anything in the corner is the credit
    for png in (render_cloud_map(_empty_query()), render_value_map(_empty_query(), 0, 7)):
        assert _corner(png).max() > 100


def test_frame_carries_the_credit_on_a_box() -> None:
    frame = Frame(np.full((300, 400, 3), 250.0), datetime.datetime(2026, 11, 13, tzinfo=datetime.UTC), saturation=255.0)
    corner = _corner(render_frame(frame))
    # a bright (white) sky: a black box with grey text in the corner
    assert corner.min() == 0
    assert ((corner > 100) & (corner < 200)).any()
