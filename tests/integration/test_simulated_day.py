"""The day pipeline on simulated daytime frames from the colour presets."""

import datetime

import numpy as np
import pytest

from pyobs_cloudcover.camera import CameraProfile, read_image
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

NOON = datetime.datetime(2026, 3, 21, 10, 0, tzinfo=datetime.UTC)  # sun at ~56 deg in Sutherland

CAMERAS = {
    "zwo-asi224mc": {},
    "rpi-hq": {"bayer": "BGGR", "bit_depth": 12},
    "budget-8bit": {"bayer": "GRBG", "orientation": {"flip_x": True}, "timestamp": {"source": "mtime"}},
}


@pytest.mark.parametrize("preset", sorted(CAMERAS))
@pytest.mark.parametrize("cover", [0.0, 0.5, 1.0])
def test_day_cloud_fraction(preset: str, cover: float, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sim = Simulator(get_preset(preset, 0.4), clouds=[CloudLayer(cover=cover, optical_depth=3, seed=4)], seed=4)
    frame = sim.render(NOON)
    path = frame.write(tmp_path)
    if CAMERAS[preset].get("timestamp", {}).get("source") == "mtime":
        import os

        os.utime(path, (NOON.timestamp(), NOON.timestamp()))
    profile = CameraProfile({**CAMERAS[preset], "world_model": sim.world_model.to_config()})
    pipelines = {"day": {"alt_interval": {"start": 18}, "options": {}}}
    controller = PipelineControllerFactory(sim._observer, profile)(pipelines)
    result = controller(profile.prepare(read_image(path)))
    assert result is not None

    measured = result.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(70))
    expected = frame.truth.cloud_fraction(20)
    assert measured is not None
    assert abs(measured - expected) < 0.15, (measured, expected)
