"""Pointed (non-fisheye) camera: calibration from stars."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from astroplan import Observer
from astropy import units as u

from pyobs_cloudcover.calibration.astrometry import catalog_altaz, detect_stars, value_step
from pyobs_cloudcover.calibration.pointed import calibrate_pointed
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.world_model import PointedModel

OBSERVER = Observer(latitude=-33.93 * u.deg, longitude=18.48 * u.deg, elevation=15 * u.m)
TIME = datetime.datetime(2026, 11, 13, 1, 30, tzinfo=datetime.UTC)
W, H = 1152, 648


def truth() -> PointedModel:
    # a Pi Camera Module 3 Wide at quarter resolution, 2 deg off the zenith
    return PointedModel(W / 2 - 3.2, H / 2 + 1.5, 488.0, rotation=np.deg2rad(97), tilt_x=0.025, tilt_y=0.023,
                        k1=0.28, k2=0.2)


def render(model: PointedModel, time: datetime.datetime, seed: int = 1) -> np.ndarray:
    """8-bit-like star field: background 60, stars as Gaussians, rounded (quantised) like a JPEG."""
    rng = np.random.default_rng(seed)
    alt, az, mag = catalog_altaz(OBSERVER, time, 5.0, 5.0)
    x, y = (np.asarray(v) for v in model.altaz_to_pix(alt, az))
    img = np.full((H, W), 60.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for xi, yi, m in zip(x, y, mag):
        if np.isfinite(xi) and 0 <= xi < W and 0 <= yi < H:
            sl = (slice(max(0, int(yi) - 5), int(yi) + 6), slice(max(0, int(xi) - 5), int(xi) + 6))
            img[sl] += 600 * 10 ** (-0.4 * (m - 1)) * np.exp(-((xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2) / 2.0)
    img += rng.normal(0, 0.3, img.shape)
    return np.asarray(np.clip(np.rint(img), 0, 255))


def test_value_step_of_quantised_images() -> None:
    assert value_step(np.arange(0, 255, 1 / 3)) == pytest.approx(1 / 3)
    assert value_step(np.random.default_rng(0).normal(size=(100, 100))) == 0.0


def test_quantised_background_does_not_make_noise_significant() -> None:
    img = np.full((200, 200), 60.0) + np.arange(200)[None, :] // 10  # smooth gradient, flat in most blocks
    img[100:103, 100:103] += 30
    x, _, _ = detect_stars(img)
    assert len(x) == 1


def test_calibrate_recovers_the_camera() -> None:
    model = truth()
    times = [TIME + datetime.timedelta(hours=h) for h in (-2, 0)]
    frames = [Frame(render(model, t, seed=i), t, saturation=255.0) for i, t in enumerate(times)]
    result = calibrate_pointed(frames, OBSERVER, hfov=100, max_tilt=6, focal_range=(0.9, 1.1))
    assert result.trustworthy
    assert result.rms < 1.0
    yy, xx = np.mgrid[20:H:60, 20:W:60].astype(float)
    alt, az = model.pix_to_altaz(xx, yy)
    x2, y2 = result.model.altaz_to_pix(alt, az)
    assert np.nanmax(np.hypot(x2 - xx, y2 - yy)) < 1.5
