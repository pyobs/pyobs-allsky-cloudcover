"""Setting up a new camera with nothing but the CLI, as docs/source/calibration.rst describes."""

import datetime
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from pyobs_cloudcover.cli import main
from pyobs_cloudcover.evaluation import evaluate
from pyobs_cloudcover.simulation import CloudLayer, Horizon, Simulator, get_preset
from pyobs_cloudcover.simulation.truth import GroundTruth

T0 = datetime.datetime(2026, 3, 20, 19, 0, tzinfo=datetime.UTC)


def write(sim: Simulator, time: datetime.datetime, directory: Path) -> Path:
    path = sim.render(time).write(directory)
    os.utime(path, (time.timestamp(), time.timestamp()))
    return path


def test_set_up_a_camera_with_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    preset = get_preset("rpi-hq")
    horizon = Horizon.trees_and_building(seed=2)

    # what the user knows: the site, and how the capture software writes files (no Bayer keyword, 12 bit)
    config = {
        "camera": {"bayer": "BGGR", "bit_depth": 12},
        "location": {"latitude": -32.38, "longitude": 20.81, "elevation": 1760.0},
        "watch": {"path": str(tmp_path), "pattern": "image-*.fits"},
        "pipelines": {"night": {"alt_interval": {"end": -18}, "options": {}}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # a few daytime frames (for the mask, and the lens circle: trees hide much of the horizon at night)
    day = tmp_path / "day"
    for hour in (7, 10, 13):
        write(
            Simulator(preset, horizon=horizon, clouds=[CloudLayer(cover=0.3, seed=hour)]),
            datetime.datetime(2026, 3, 21, hour, tzinfo=datetime.UTC),
            day,
        )

    # 1. world model from two clear nights frames
    clear = tmp_path / "clear"
    sim = Simulator(preset, horizon=horizon, seed=1)
    images = [str(write(sim, T0 + datetime.timedelta(hours=h), clear)) for h in (1, 3)]
    assert main(["calibrate", str(config_path), *images, "--circle-from", str(day)]) == 0
    fitted = yaml.safe_load(capsys.readouterr().out)["world_model"]
    config["camera"]["world_model"] = fitted
    config_path.write_text(yaml.safe_dump(config))

    # 2. sky mask from the daytime frames
    assert main(["mask", str(config_path), str(day), "-o", str(tmp_path / "mask.npy")]) == 0

    # 3. hot pixels from night frames
    night = tmp_path / "night"
    for i in range(4):
        write(sim, T0 + datetime.timedelta(hours=1, minutes=40 * i), night)
    assert main(["hot-pixels", str(config_path), str(night), "-o", str(tmp_path / "hot.npy")]) == 0
    config["camera"].update(mask=str(tmp_path / "mask.npy"), hot_pixels=str(tmp_path / "hot.npy"))
    config_path.write_text(yaml.safe_dump(config))

    # 4. a night with changing clouds: replay suggests a threshold (from the data, no truth needed)
    archive = tmp_path / "archive"
    for i, cover in enumerate((0.0, 0.2, 0.5, 0.8, 1.0, 0.4)):
        cloudy = Simulator(preset, horizon=horizon, clouds=[CloudLayer(cover=cover, seed=i)], seed=i)
        path = write(cloudy, T0 + datetime.timedelta(hours=1, minutes=30 * i), archive)
        GroundTruth.path_for(path).unlink()  # a real archive has no ground truth
    capsys.readouterr()
    assert main(["replay", str(config_path), str(archive), "--csv", str(tmp_path / "replay.csv")]) == 0
    out = capsys.readouterr().out
    threshold = float(out.split("suggested night threshold: ")[1].split()[0])
    assert 3.0 < threshold < 6.0
    assert (tmp_path / "replay.csv").read_text().count("\n") == 7

    # the finished config analyses a new frame well
    config["pipelines"]["night"]["options"]["cloud_map"] = {"threshold": threshold}
    from pyobs_cloudcover.evaluation import Scenario, run_scenario

    info, truth = run_scenario(Scenario("rpi-hq", "dark", 0.5, seed=7), config, tmp_path / "check")
    metrics = evaluate(info, truth)
    assert metrics is not None and metrics.agreement > 0.75, metrics
    # the fitted model matches the real lens
    true = sim.world_model
    alt, az = np.meshgrid(np.deg2rad(np.linspace(20, 89, 10)), np.deg2rad(np.arange(0, 360, 30)))
    from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory

    model = WorldModelFactory(sim._observer)(fitted)
    err = np.hypot(
        *(np.asarray(a) - np.asarray(b) for a, b in zip(true.altaz_to_pix(alt, az), model.altaz_to_pix(alt, az)))
    )
    assert np.nanmax(err) < 1.0
