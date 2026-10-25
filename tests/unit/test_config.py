"""The shipped example config and a 1.x config both build a working module."""

from pathlib import Path
from typing import Any

import yaml

from pyobs_cloudcover.application import Application
from pyobs_cloudcover.measurement_log.logger_strategies.memory import MemoryLog

ROOT = Path(__file__).parents[2]


def build(config: dict[str, Any]) -> Application:
    config = dict(config)
    config.pop("class")
    config["measurement_log"]["logger"] = {"type": "memory"}
    location = config.pop("location", {"longitude": 0.0, "latitude": 0.0, "elevation": 0.0})
    return Application(**config, location=location)


def test_example_config(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "example.yaml").read_text())
    config["watch"]["path"] = str(tmp_path)
    app = build(config)
    assert isinstance(app._measurement_log.strategy, MemoryLog)
    assert app._watcher is not None


def test_1x_config_still_works() -> None:
    # example.yaml as shipped with 1.2.1 (only the deleted mask files removed)
    config = yaml.safe_load((ROOT / "tests" / "data" / "example_1x.yaml").read_text())
    app = build(config)
    assert app._image_sender == "allskycam"
    night = app._pipeline_controller._pipelines[1]
    # 1.x semantics: vertical flip, 1.x detector and estimator
    assert night._preprocess._flip_vertical
    assert night._lim_magnitude_map_generator._estimate.__name__ == "border_magnitude"
