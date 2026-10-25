"""Compare analysis results with the simulator's ground truth."""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.simulation.truth import GroundTruth


@dataclass
class Metrics:
    """Accuracy of one result on the positions above ``min_alt`` where it has a value.

    Fractions are of those positions (the grid is equal-area, so they are sky-area fractions); the true
    fractions use the same positions, so masked areas (moon, sun, obstructions) don't count against a result.
    """

    agreement: float
    total_measured: float
    total_true: float
    zenith_measured: float | None
    zenith_true: float | None
    coverage: float  # fraction of positions above min_alt with a value

    @property
    def total_error(self) -> float:
        return abs(self.total_measured - self.total_true)

    @property
    def zenith_error(self) -> float | None:
        if self.zenith_measured is None or self.zenith_true is None:
            return None
        return abs(self.zenith_measured - self.zenith_true)


def _positions(info: CloudCoverageInfo, min_alt: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    grid = info.cloud_cover_query.grid
    return np.rad2deg(grid.alt), np.rad2deg(grid.az)


def evaluate(
    info: CloudCoverageInfo,
    truth: GroundTruth,
    min_alt: float = 30.0,
    zenith_alt: float = 80.0,
    threshold: float | None = None,
) -> Metrics | None:
    """Score a result. With ``threshold``, the cloud map is re-derived from the continuous map (``value <
    threshold`` = cloudy), which lets thresholds be tuned without re-running the pipeline."""
    alt, az = _positions(info, min_alt)
    if threshold is None:
        cloudy_map = info.cloud_cover_query.get_pixels()
    else:
        if info.value_query is None:
            raise ValueError("Result has no continuous map to threshold")
        values = info.value_query.get_pixels()
        cloudy_map = np.where(np.isnan(values), np.nan, (values < threshold).astype(np.float64))
    above = alt >= min_alt
    known = above & ~np.isnan(cloudy_map)
    if not np.any(known):
        return None
    measured = cloudy_map[known] > 0.5
    true = truth.cloudy_at(alt[known], az[known])
    zenith = alt[known] >= zenith_alt
    return Metrics(
        agreement=float(np.mean(measured == true)),
        total_measured=float(np.mean(measured)),
        total_true=float(np.mean(true)),
        zenith_measured=float(np.mean(measured[zenith])) if np.any(zenith) else None,
        zenith_true=float(np.mean(true[zenith])) if np.any(zenith) else None,
        coverage=float(np.sum(known)) / max(1.0, float(np.sum(above))),
    )


def best_threshold(
    results: Sequence[tuple[CloudCoverageInfo, GroundTruth]],
    candidates: npt.ArrayLike,
    min_alt: float = 30.0,
) -> tuple[float, float]:
    """The threshold with the best mean agreement over several results, and that agreement."""
    best, best_score = float("nan"), -1.0
    for t in np.asarray(candidates, dtype=np.float64):
        scores = [m.agreement for info, truth in results if (m := evaluate(info, truth, min_alt, threshold=t))]
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score:
            best, best_score = float(t), score
    return best, best_score


# ---------------------------------------------------------------------------------------------- scenarios

NIGHTS = {
    # Sutherland (the simulator's default site)
    "dark": datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC),  # moon down
    "quarter moon": datetime.datetime(2026, 4, 10, 1, 0, tzinfo=datetime.UTC),  # 52 %, 40 deg up
    "full moon": datetime.datetime(2026, 3, 30, 22, 0, tzinfo=datetime.UTC),  # 95 %, 49 deg up
}
DAY = datetime.datetime(2026, 3, 21, 10, 0, tzinfo=datetime.UTC)  # sun 56 deg up


@dataclass(frozen=True)
class Scenario:
    preset: str
    sky: str  # "dark", "quarter moon", "full moon" or "day"
    cover: float
    optical_depth: float = 3.0
    seed: int = 0

    @property
    def time(self) -> datetime.datetime:
        return DAY if self.sky == "day" else NIGHTS[self.sky]

    @property
    def label(self) -> str:
        return f"{self.preset} {self.sky} cover={self.cover:g}"


def scenario_matrix(
    presets: Sequence[str], covers: Sequence[float] = (0.0, 0.4, 1.0), seeds: Sequence[int] = (0,)
) -> list[Scenario]:
    from pyobs_cloudcover.simulation import get_preset

    scenarios = []
    for preset in presets:
        skies = [*NIGHTS, "day"] if get_preset(preset).is_color else list(NIGHTS)
        for sky in skies:
            for cover in covers:
                for seed in seeds:
                    scenarios.append(Scenario(preset, sky, cover, seed=seed))
    return scenarios


def run_scenario(
    scenario: Scenario, config: dict[str, Any], directory: str | Path, scale: float = 1.0
) -> tuple[CloudCoverageInfo | None, GroundTruth]:
    """Simulate one frame, write it to ``directory``, and analyse it with the ``camera`` and ``pipelines``
    sections of a module config, as the module would."""
    from pyobs_cloudcover.camera import CameraProfile, read_image
    from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
    from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

    clouds = [CloudLayer(cover=scenario.cover, optical_depth=scenario.optical_depth, seed=scenario.seed + 100)]
    sim = Simulator(get_preset(scenario.preset, scale), clouds=clouds, seed=scenario.seed)
    frame = sim.render(scenario.time)
    path = frame.write(directory)
    os.utime(path, (scenario.time.timestamp(), scenario.time.timestamp()))

    camera = dict(config.get("camera") or {})
    if scale != 1.0:
        camera["world_model"] = sim.world_model.to_config()
    profile = CameraProfile(camera)
    controller = PipelineControllerFactory(sim._observer, profile)(config["pipelines"])
    return controller(profile.prepare(read_image(path))), frame.truth
