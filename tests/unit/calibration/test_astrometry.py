import datetime
import os
from pathlib import Path

import numpy as np
import pytest

from pyobs_cloudcover.calibration.astrometry import calibrate, fit_lens_circle, luminance
from pyobs_cloudcover.camera import CameraProfile, Frame, read_image
from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

CAMERAS = {
    "zwo-asi224mc": {},
    "rpi-hq": {"bayer": "BGGR", "bit_depth": 12},
    "zwo-asi178mm": {"orientation": {"rotate": 90}},
    "budget-8bit": {"bayer": "GRBG", "orientation": {"flip_x": True}, "timestamp": {"source": "mtime"}},
}


def frames(preset: str, tmp_path: Path, clouds: list[CloudLayer] | None = None) -> tuple[Simulator, list[Frame]]:
    sim = Simulator(get_preset(preset), clouds=clouds or [], seed=3)
    profile = CameraProfile(CAMERAS[preset])
    out = []
    for hour in (20, 22):
        t = datetime.datetime(2026, 3, 20, hour, 0, tzinfo=datetime.UTC)
        path = sim.render(t).write(tmp_path)
        os.utime(path, (t.timestamp(), t.timestamp()))
        out.append(profile.prepare(read_image(path)))
    return sim, out


def max_error(true, fitted, min_alt: float) -> float:  # type: ignore[no-untyped-def]
    alt, az = np.meshgrid(np.deg2rad(np.linspace(min_alt, 89, 25)), np.deg2rad(np.arange(0, 360, 10)))
    x1, y1 = true.altaz_to_pix(alt, az)
    x2, y2 = fitted.altaz_to_pix(alt, az)
    return float(np.nanmax(np.hypot(np.asarray(x1) - x2, np.asarray(y1) - y2)))


@pytest.mark.parametrize("preset", ["zwo-asi224mc", "rpi-hq", "zwo-asi178mm"])
def test_recovers_the_world_model(preset: str, tmp_path: Path) -> None:
    sim, fs = frames(preset, tmp_path)
    result = calibrate(fs, sim._observer)
    assert result.trustworthy
    assert result.lowest_altitude <= 20
    assert result.rms < 1.0
    # lens, mirror parity, rotation and scale from nothing but site and time
    assert max_error(sim.world_model, result.model, 20) < 0.5
    np.testing.assert_allclose(np.rad2deg(result.model.a0) % 360, np.rad2deg(sim.world_model.a0) % 360, atol=0.1)


def test_cheap_lens_is_approximated_where_stars_are_seen(tmp_path: Path) -> None:
    sim, fs = frames("budget-8bit", tmp_path)
    result = calibrate(fs, sim._observer)
    assert result.trustworthy
    assert result.lowest_altitude <= 45
    assert max_error(sim.world_model, result.model, result.lowest_altitude) < 8


def test_overcast_is_not_trusted(tmp_path: Path) -> None:
    sim, fs = frames("zwo-asi224mc", tmp_path, [CloudLayer(cover=1.0, optical_depth=5)])
    result = calibrate(fs, sim._observer)
    assert not result.trustworthy


def test_lens_circle(tmp_path: Path) -> None:
    sim, fs = frames("zwo-asi224mc", tmp_path)
    circle = fit_lens_circle([luminance(f) for f in fs])
    assert abs(circle.c_x - sim.world_model.c_x) < 1 and abs(circle.c_y - sim.world_model.c_y) < 1
    # the ground below the horizon is nearly black in the simulation, so the disc ends at the horizon
    expected = float(sim.world_model.radius(np.deg2rad(90.0)))
    assert abs(circle.radius - expected) < 3
