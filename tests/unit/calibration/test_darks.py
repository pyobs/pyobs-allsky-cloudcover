import datetime
from pathlib import Path

import numpy as np

from pyobs_cloudcover.calibration.darks import (
    hot_pixels_from_dark,
    hot_pixels_from_frames,
    master_dark,
    write_dark,
)
from pyobs_cloudcover.camera import CameraProfile, read_image
from pyobs_cloudcover.camera.reader import RawImage
from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.simulation import Simulator, get_preset

NIGHT = datetime.datetime(2026, 3, 20, 20, 0, tzinfo=datetime.UTC)


def strong_hot_pixels(sim: Simulator, exposure: float) -> np.ndarray:
    """Simulated hot pixels that are clearly above the noise in one exposure."""
    y, x, rate = sim.hot_pixels
    p = sim.profile
    strong = rate * exposure > 30 * p.read_noise
    mask = np.zeros((p.height, p.width), bool)
    mask[y[strong], x[strong]] = True
    return mask


def test_hot_pixels_from_night_frames(tmp_path: Path) -> None:
    sim = Simulator(get_preset("budget-8bit", 0.5), seed=1)
    frames = [read_image(sim.render(NIGHT + datetime.timedelta(minutes=40 * i)).write(tmp_path)) for i in range(5)]
    found = hot_pixels_from_frames(frames)
    true = strong_hot_pixels(sim, sim.profile.exposure_time)
    assert true.sum() > 20
    assert np.mean(found[true]) > 0.9  # recall
    assert found.sum() < 1.5 * true.sum() + 10  # hardly any stars or noise


def test_master_dark_and_hot_pixels(tmp_path: Path) -> None:
    sim = Simulator(get_preset("zwo-asi224mc", 0.5), seed=1)
    darks = [read_image(sim.render_dark(NIGHT + datetime.timedelta(minutes=i)).write(tmp_path)) for i in range(5)]
    dark, exposure = master_dark(darks)
    assert exposure == sim.profile.exposure_time
    hot = hot_pixels_from_dark(dark)
    true = strong_hot_pixels(sim, exposure)
    assert np.mean(hot[true]) > 0.9 and hot.sum() < 1.5 * true.sum() + 10

    write_dark(tmp_path / "dark.fits", dark, exposure)
    np.save(tmp_path / "hot.npy", hot)
    # with the dark subtracted, a night frame's background is flat at the sky level
    frame = sim.render(NIGHT)
    calibrated = CameraProfile({"dark": str(tmp_path / "dark.fits")}).prepare(read_image(frame.write(tmp_path)))
    assert abs(float(np.median(calibrated.data))) < float(np.median(frame.raw))


def test_hot_pixels_no_longer_look_like_stars(tmp_path: Path) -> None:
    """Catalogue stars predicted on a hot pixel are 'seen' unless the hot-pixel map is applied."""
    sim = Simulator(get_preset("budget-8bit", 0.5), seed=1, faint_star_limit=0)
    frames = [sim.render(NIGHT + datetime.timedelta(minutes=40 * i)) for i in range(5)]
    paths = [f.write(tmp_path) for f in frames]
    hot = hot_pixels_from_frames([read_image(p) for p in paths])
    np.save(tmp_path / "hot.npy", hot)

    # pretend a catalogue star sits on every strong hot pixel (mono-style matching on the raw frame)
    y, x = np.nonzero(strong_hot_pixels(sim, sim.profile.exposure_time))
    inside = (x > 10) & (x < sim.profile.width - 10) & (y > 10) & (y < sim.profile.height - 10)
    y, x = y[inside], x[inside]
    zeros = np.zeros(len(x))
    catalog = PixelCatalog(np.arange(len(x)), zeros, zeros, x.astype(float), y.astype(float), zeros)
    matcher = ApertureMatcher(snr=3)

    raw = RawImage(frames[0].raw, header={"DATE-OBS": "2026-03-20T20:00:00"})
    without = CameraProfile({"bayer": "none"}).prepare(raw).data
    with_map = CameraProfile({"bayer": "none", "hot_pixels": str(tmp_path / "hot.npy")}).prepare(raw).data
    assert np.mean(matcher(without, catalog)) > 0.7
    assert np.mean(matcher(with_map, catalog)) < 0.2
