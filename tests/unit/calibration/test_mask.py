import datetime
from pathlib import Path

import numpy as np

from pyobs_cloudcover.calibration.mask import sky_mask_from_altitude, sky_mask_from_day
from pyobs_cloudcover.camera import CameraProfile, read_image
from pyobs_cloudcover.simulation import CloudLayer, Horizon, Simulator, get_preset


def test_mask_finds_trees_and_buildings(tmp_path: Path) -> None:
    horizon = Horizon.trees_and_building(seed=4)
    profile = CameraProfile()
    frames = []
    for hour, cover in ((8, 0.0), (10, 0.5), (13, 0.0)):
        sim = Simulator(get_preset("zwo-asi224mc", 0.5), horizon=horizon, clouds=[CloudLayer(cover=cover, seed=hour)])
        t = datetime.datetime(2026, 3, 21, hour, 0, tzinfo=datetime.UTC)
        frames.append(profile.prepare(read_image(sim.render(t).write(tmp_path))))
    model = sim.world_model

    mask = sky_mask_from_day(frames, model)

    # what the simulator actually obstructed
    h, w = mask.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    alt, az = model.pix_to_altaz(x, y)
    alt, az = np.asarray(alt), np.asarray(az)
    valid = np.isfinite(alt) & (alt > 0)
    truth = valid & ~horizon.is_obstructed(np.where(valid, alt, 0), np.where(valid, az, 0))

    obstructed = valid & ~truth
    assert obstructed.sum() > 1000  # the building and the trees are big enough to matter
    # most obstructed pixels are masked, almost no sky is lost
    assert np.mean(~mask[obstructed]) > 0.85
    assert np.mean(mask[truth]) > 0.97
    # the building (22 deg high around azimuth 30-70) is masked
    building = valid & (np.rad2deg(az) > 35) & (np.rad2deg(az) < 65) & (np.rad2deg(alt) < 20)
    assert np.mean(~mask[building]) > 0.85


def test_altitude_mask() -> None:
    model = Simulator(get_preset("zwo-asi224mc", 0.2)).world_model
    mask = sky_mask_from_altitude((196, 262), model, 30.0)
    y, x = np.nonzero(mask)
    alt, _ = model.pix_to_altaz(x.astype(float), y.astype(float))
    assert np.all(np.rad2deg(alt) >= 30)
    assert 0.1 < mask.mean() < 0.6
