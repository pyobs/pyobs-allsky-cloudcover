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
        pipelines = {"night": {"alt_interval": {"end": -18}, "options": {"cloud_map": {"threshold": threshold}}}}
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
