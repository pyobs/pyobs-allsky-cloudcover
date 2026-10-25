"""Run the pipelines over stored images: a CSV of results and help choosing thresholds."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astroplan import Observer

from pyobs_cloudcover.camera import CameraProfile, read_image
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.evaluation import best_threshold, evaluate
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.simulation.truth import GroundTruth

log = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    path: str
    info: CloudCoverageInfo | None
    truth: GroundTruth | None = None
    error: str = ""


def replay(
    config: dict[str, Any],
    paths: Sequence[str | Path],
    observer: Observer,
    progress: Callable[[ReplayResult], None] | None = None,
) -> list[ReplayResult]:
    """Analyse images (sorted by name) with the ``camera`` and ``pipelines`` sections of a module config. Ground
    truth next to simulated images (``.truth.npz``) is picked up for scoring."""
    profile = CameraProfile(config.get("camera"))
    controller = PipelineControllerFactory(observer, profile)(config["pipelines"])
    results = []
    for path in sorted(map(str, paths)):
        truth_path = GroundTruth.path_for(path)
        truth = GroundTruth.load(truth_path) if truth_path.exists() else None
        try:
            info = controller(profile.prepare(read_image(path)))
            result = ReplayResult(path, info, truth)
        except Exception as e:  # keep going, report it
            result = ReplayResult(path, None, truth, f"{type(e).__name__}: {e}")
        results.append(result)
        if progress:
            progress(result)
    return results


def otsu_threshold(values: npt.ArrayLike, bins: int = 200) -> float:
    """The cut that best separates two populations (Otsu's method)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        raise ValueError("Too few values")
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centres)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return float(centres[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))])


@dataclass
class ThresholdSuggestion:
    pipeline: str
    threshold: float
    method: str  # "ground truth" or "otsu"
    frames: int
    agreement: float | None = None  # vs ground truth, if there is any


def suggest_thresholds(results: Sequence[ReplayResult], min_alt: float = 30.0) -> list[ThresholdSuggestion]:
    """Per pipeline: the best threshold against ground truth if the images have it, otherwise the value that best
    separates the two populations in the pooled continuous maps (needs a mix of clear and cloudy frames)."""
    suggestions = []
    by_pipeline: dict[str, list[ReplayResult]] = {}
    for r in results:
        if r.info is not None and r.info.value_query is not None:
            by_pipeline.setdefault(r.info.pipeline, []).append(r)
    for pipeline, rs in sorted(by_pipeline.items()):
        with_truth = [(r.info, r.truth) for r in rs if r.truth is not None and r.info is not None]
        values = np.concatenate(
            [
                r.info.value_query.get_pixels()[np.rad2deg(r.info.value_query.grid.alt) >= min_alt]
                for r in rs
                if r.info is not None and r.info.value_query is not None
            ]
        )
        if len(with_truth) >= 3:
            lo, hi = np.nanpercentile(values, [1, 99])
            t, score = best_threshold(with_truth, np.linspace(lo, hi, 200), min_alt)
            suggestions.append(ThresholdSuggestion(pipeline, t, "ground truth", len(rs), score))
        else:
            t = otsu_threshold(values)
            suggestions.append(ThresholdSuggestion(pipeline, t, "otsu", len(rs)))
    return suggestions


def write_csv(results: Sequence[ReplayResult], path: str | Path, min_alt: float = 30.0) -> None:
    fields = [
        "file",
        "time",
        "pipeline",
        "total",
        "zenith",
        "change",
        "value_median",
        "true_total",
        "agreement",
        "error",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row: dict[str, Any] = {"file": Path(r.path).name, "error": r.error}
            if r.info is not None:
                q = r.info.cloud_cover_query
                row.update(
                    time=r.info.obs_time.isoformat(),
                    pipeline=r.info.pipeline,
                    total=q.query_radius(np.pi / 2, 0.0, np.deg2rad(90 - min_alt)),
                    zenith=q.query_radius(np.pi / 2, 0.0, np.deg2rad(10)),
                    change=r.info.change,
                )
                if r.info.value_query is not None:
                    v = r.info.value_query.get_pixels()
                    row["value_median"] = float(np.nanmedian(v)) if np.any(np.isfinite(v)) else None
                if r.truth is not None and (m := evaluate(r.info, r.truth, min_alt)) is not None:
                    row.update(true_total=m.total_true, agreement=m.agreement)
            w.writerow(row)


def plot(results: Sequence[ReplayResult], suggestions: Sequence[ThresholdSuggestion], path: str | Path) -> None:
    """Time series of the cloud fractions and histograms of the continuous maps (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.info is not None]
    pipelines = sorted({r.info.pipeline for r in ok if r.info is not None})
    fig, axes = plt.subplots(1 + len(pipelines), 1, figsize=(10, 3 + 3 * len(pipelines)))
    axes = np.atleast_1d(axes)
    times = [r.info.obs_time for r in ok if r.info is not None]
    total = [r.info.cloud_cover_query.query_radius(np.pi / 2, 0.0, np.deg2rad(60)) for r in ok if r.info]
    axes[0].plot(times, [np.nan if t is None else t for t in total], ".-", label="measured")
    if any(r.truth for r in ok):
        axes[0].plot(times, [r.truth.cloud_fraction(30) if r.truth else np.nan for r in ok], ".", label="true")
    axes[0].set_ylabel("cloud fraction (alt > 30)")
    axes[0].legend()
    for ax, pipeline in zip(axes[1:], pipelines):
        values = np.concatenate(
            [
                r.info.value_query.get_pixels()
                for r in ok
                if r.info and r.info.pipeline == pipeline and r.info.value_query
            ]
        )
        ax.hist(values[np.isfinite(values)], bins=100)
        for s in suggestions:
            if s.pipeline == pipeline:
                ax.axvline(s.threshold, color="r", label=f"suggested {s.threshold:.2f} ({s.method})")
        ax.set_xlabel(f"{pipeline}: {ok[0].info.value_name if ok[0].info else ''}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
