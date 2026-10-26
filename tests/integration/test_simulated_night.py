"""The night pipeline on simulated frames from every camera preset, configured as a user would."""

import datetime

import numpy as np
import pytest

from pyobs_cloudcover.camera import CameraProfile, read_image
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset
from pyobs_cloudcover.world_model import FisheyeModel

NIGHT = datetime.datetime(2026, 3, 22, 22, 0, tzinfo=datetime.UTC)
SCALE = 0.5

CAMERAS = {
    "zwo-asi224mc": {},
    "rpi-hq": {"bayer": "BGGR", "bit_depth": 12},
    "zwo-asi178mm": {"orientation": {"rotate": 90}},
}
# the limiting-magnitude threshold is camera specific; these are the ones benchmarks/night_method.py trained
THRESHOLDS = {"zwo-asi224mc": 4.05, "rpi-hq": 4.8, "zwo-asi178mm": 5.25}


def run(preset: str, cover: float, tmp_path, camera: dict | None = None, pipelines: dict | None = None):  # type: ignore[no-untyped-def]
    sim = Simulator(get_preset(preset, SCALE), clouds=[CloudLayer(cover=cover, optical_depth=4, seed=2)], seed=2)
    frame = sim.render(NIGHT)
    path = frame.write(tmp_path)
    if camera is None:
        camera = {**CAMERAS[preset], "world_model": sim.world_model.to_config()}
    profile = CameraProfile(camera)
    if pipelines is None:
        threshold = THRESHOLDS[preset]
        # the limiting-magnitude cloud map; the photometric one needs full-size images (test_photometric_mode)
        options = {"cloud_map": {"threshold": threshold}, "photometry": {"cloud_map": False}}
        pipelines = {"night": {"alt_interval": {"end": -18}, "options": options}}
    controller = PipelineControllerFactory(sim._observer, profile)(pipelines)
    result = controller(profile.prepare(read_image(path)))
    assert result is not None
    return result, frame.truth


@pytest.mark.parametrize("preset", sorted(CAMERAS))
def test_clear_and_overcast(preset: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    clear, _ = run(preset, 0.0, tmp_path / "clear")
    overcast, truth = run(preset, 1.0, tmp_path / "overcast")
    assert truth.cloud_fraction(30) == 1.0
    total_clear = clear.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(60))
    total_overcast = overcast.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(60))
    assert total_clear is not None and total_overcast is not None
    assert total_clear < 0.3
    assert total_overcast > 0.7


def test_legacy_config_still_works(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A 1.x config: model fitted on the vertically flipped, 2x2-binned image, 1.x detector and estimator."""
    sim = Simulator(get_preset("zwo-asi224mc", SCALE))
    model = sim.world_model.scaled(0.5)
    binned_height = sim.profile.canonical_shape[0] // 2
    # flipping the image vertically mirrors the model: y -> H-1-y, i.e. the opposite parity
    flipped = FisheyeModel(
        model.c_x, binned_height - 1 - model.c_y, model.f, model.a0, model.projection, parity=-model.parity
    )
    legacy = {
        "night": {
            "alt_interval": {"start": None, "end": -18},
            "options": {
                "world_model": flipped.to_config(),
                "preprocessor": {"mask_filepath": "", "bin_size": 2},
                "catalog": {"filepath": "data/catalog_v50.csv", "filter": {"alt": 30.0, "v_mag": 7.0, "distance": 0.0}},
                "reverse_matcher": {"sigma_threshold": 3.0, "window_size": 6, "median_limit": 1e9},
                "altaz_grid": {"point_number": 100000, "limiting_altitude": 30},
                "lim_mag_map": {"radius": 7.0, "integration_length": 1},
                "cloud_map": {"threshold": 5.5},
                "moon_apparent_size": 0.5,
            },
        }
    }
    clear, _ = run("zwo-asi224mc", 0.0, tmp_path / "clear", camera={}, pipelines=legacy)
    overcast, _ = run("zwo-asi224mc", 1.0, tmp_path / "overcast", camera={}, pipelines=legacy)
    total_clear = clear.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(60))
    total_overcast = overcast.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(60))
    assert total_clear is not None and total_overcast is not None
    assert total_overcast > total_clear + 0.3


def test_mask_keeps_obstructions_from_reading_as_cloud(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pyobs_cloudcover.calibration.mask import sky_mask_from_altitude
    from pyobs_cloudcover.simulation import Horizon

    # a wall up to 45 deg all around the east
    horizon = Horizon(azimuth=(0.0, 10.0, 170.0, 180.0, 359.0), altitude=(0.0, 45.0, 45.0, 0.0, 0.0))
    sim = Simulator(get_preset("zwo-asi178mm", SCALE), horizon=horizon, seed=2)
    path = sim.render(NIGHT).write(tmp_path)
    camera = {**CAMERAS["zwo-asi178mm"], "world_model": sim.world_model.to_config()}

    # sky mask: above 46 deg in the east, above 30 deg elsewhere (as the mask builder would produce)
    model = sim.world_model
    h, w = sim.profile.canonical_shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    _, az = model.pix_to_altaz(x, y)
    east = (np.rad2deg(np.asarray(az)) > 5) & (np.rad2deg(np.asarray(az)) < 175)
    mask = sky_mask_from_altitude((h, w), model, 30.0) & (~east | sky_mask_from_altitude((h, w), model, 46.0))
    np.save(tmp_path / "mask.npy", mask)

    options = {"cloud_map": {"threshold": 4.8}, "photometry": {"cloud_map": False}}
    pipelines = {"night": {"alt_interval": {"end": -18}, "options": options}}
    results = {}
    for name, cam in (("without", camera), ("with", {**camera, "mask": str(tmp_path / "mask.npy")})):
        profile = CameraProfile(cam)
        controller = PipelineControllerFactory(sim._observer, profile)(pipelines)
        result = controller(profile.prepare(read_image(path)))
        assert result is not None
        # cloud fraction in the obstructed band (east, 30-45 deg)
        results[name] = result.cloud_cover_query.query_radius(np.deg2rad(37), np.deg2rad(90), np.deg2rad(7))
    assert results["without"] is not None and results["without"] > 0.7  # the wall looks like cloud
    assert results["with"] is None or results["with"] < 0.3  # masked: no stars there, so no claim


def test_photometric_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Full size: cloud extinction per position and the zenith sky brightness. Where stars measure the extinction,
    a full moon doesn't fool it and thin cirrus is seen; elsewhere the limiting magnitude fills in, so the combined
    map beats the limiting-magnitude map alone."""
    from pyobs_cloudcover.evaluation import evaluate

    dark = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)
    full_moon = datetime.datetime(2026, 3, 30, 22, 0, tzinfo=datetime.UTC)
    threshold = THRESHOLDS["zwo-asi224mc"]
    pipelines = {"night": {"alt_interval": {"end": -18}, "options": {"cloud_map": {"threshold": threshold}}}}
    profile = None
    controller = None
    # the zero point is learnt on a dark, clear night; then a full moon, clear and with thin cirrus
    scenes = (
        ("dark", dark, None),
        ("clear", full_moon, None),
        ("cirrus", full_moon, CloudLayer(cover=0.6, optical_depth=0.8, seed=3)),
    )
    for name, time, layer in scenes:
        sim = Simulator(get_preset("zwo-asi224mc"), clouds=[layer] if layer else [], seed=3)
        frame = sim.render(time)
        if controller is None:
            profile = CameraProfile({"world_model": sim.world_model.to_config()})
            controller = PipelineControllerFactory(sim._observer, profile)(pipelines)
        assert profile is not None
        info = controller(profile.prepare(read_image(frame.write(tmp_path / name))))
        assert info is not None and info.extinction_query is not None

        # where stars measure it, the extinction is right
        q = info.extinction_query
        alt, az = np.rad2deg(q.grid.alt), np.rad2deg(q.grid.az)
        ext = q.get_pixels()
        measured = np.isfinite(ext) & (alt >= 30)
        assert measured.mean() > 0.05, name  # under a full moon only a small part of the sky is measured
        true_cloudy = frame.truth.cloudy_at(alt[measured], az[measured])
        assert np.mean((ext[measured] >= 0.5) == true_cloudy) > 0.65, name

        # the combined cloud map beats the limiting magnitude alone
        combined = evaluate(info, frame.truth)
        lim_only = evaluate(info, frame.truth, threshold=threshold)
        assert combined is not None and lim_only is not None
        assert combined.agreement >= lim_only.agreement - 0.01, (name, combined, lim_only)
        if name == "clear":
            assert abs(info.sky_brightness - frame.truth.metadata["sky_zenith_mag"]) < 0.2
