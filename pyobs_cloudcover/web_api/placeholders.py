"""Why an image route can't show a real image right now, as a short title and an explanation.

The image routes return a placeholder image with this text instead of an error, so dashboards that embed them
(e.g. pyobs-weather's Allsky page) say what is going on: daytime analysis switched off, camera not delivering
images, analysis failed, and so on. For an expected gap (e.g. no cloud map in daylight when only the night
pipeline is configured) they name the pipeline they are waiting for and its configured sun altitude.
``detail`` may contain line breaks (paragraphs).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Literal

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.schedule import AnalysisSchedule, describe_interval

Kind = Literal["image", "sky", "cloud_map", "limiting_magnitude", "extinction"]
FRAME_KINDS = ("image", "sky")


@dataclass(frozen=True)
class Placeholder:
    title: str
    detail: str


def _ago(then: datetime.datetime, now: datetime.datetime) -> str:
    s = max(0.0, (now - then).total_seconds())
    if s < 90:
        return f"{s:.0f} s ago"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h ago"
    return f"{s / 86400:.1f} d ago"


def _ut(t: datetime.datetime) -> str:
    return t.astimezone(datetime.UTC).strftime("%Y-%m-%d %H:%M UT")


def _sun_from(reason: str) -> float | None:
    """The sun altitude from a 'no pipeline for the sun at X deg' reason."""
    match = re.search(r"sun at (-?[\d.]+) deg", reason)
    return float(match.group(1)) if match else None


def _waiting_for_dark(schedule: AnalysisSchedule | None) -> Placeholder:
    """For the maps that only the night pipeline makes (limiting magnitude, extinction)."""
    night = schedule.window("night") if schedule is not None else None
    if night is None:
        return Placeholder("Only at night", "No night pipeline is configured.")
    title = "Waiting for astronomical dark" if night.end == -18 else "Waiting for night"
    return Placeholder(title, f"Night pipeline: {describe_interval(night)}.")


def _no_pipeline(sun: float, schedule: AnalysisSchedule | None) -> Placeholder:
    """For the cloud map when no pipeline covers the sun's altitude."""
    night = schedule.window("night") if schedule is not None else None
    if sun >= 0:
        title = "Daytime pipeline is off"
    elif sun > -18:
        title = "No twilight pipeline"
    else:
        title = "No pipeline for this sun altitude"
    if night is not None and sun not in night:
        return Placeholder(title, f"Waiting for night: {describe_interval(night)}.")
    return Placeholder(title, f"Sun at {sun:.0f}\N{DEGREE SIGN}.")


def placeholder_reason(
    kind: Kind,
    now: datetime.datetime,
    stale_after: float,
    frame_time: datetime.datetime | None,
    measurement: CloudCoverageInfo | None,
    last_reason: str = "",
    last_error: str = "",
    last_error_at: datetime.datetime | None = None,
    has_world_model: bool = True,
    schedule: AnalysisSchedule | None = None,
) -> Placeholder | None:
    """None if the route can show its real image, else why not.

    ``frame_time``: observation time of the latest analysed image; ``measurement``: the latest result;
    ``last_reason``: why the latest image gave no result; ``last_error(_at)``: the latest failure; ``schedule``:
    the pipelines' sun-altitude windows (to say what an expected gap is waiting for)."""
    # the camera (or the capture software, or the hand-over to this module) has stopped
    if frame_time is None and measurement is None:
        return Placeholder("No image yet", "Waiting for the first image from the camera.")
    newest = max(t for t in (frame_time, measurement.obs_time if measurement else None) if t is not None)
    if (now - newest).total_seconds() > stale_after:
        return Placeholder(
            "No new image",
            f"Last image {_ut(newest)} ({_ago(newest, now)}). Is the camera or the capture software running?",
        )
    if last_error_at is not None and last_error_at > newest:
        return Placeholder("Analysis failed", last_error or "The latest image could not be analysed.")

    if kind in FRAME_KINDS:
        if frame_time is None:
            return Placeholder("No image yet", "Waiting for the first image from the camera.")
        if kind == "sky" and not has_world_model:
            return Placeholder("No north-up image", "No world model is configured (camera.world_model).")
        return None

    # maps: the latest image gave no result (e.g. no pipeline for the sun altitude)
    night_map = kind in ("limiting_magnitude", "extinction")
    if measurement is None or (frame_time is not None and frame_time > measurement.obs_time):
        sun = _sun_from(last_reason)
        if sun is None:
            if last_reason:
                return Placeholder("No analysis for this image", last_reason)
            return Placeholder("No analysis yet", "Waiting for the first result.")
        if night_map:
            return _waiting_for_dark(schedule)
        return _no_pipeline(sun, schedule)
    if kind == "limiting_magnitude" and (
        measurement.value_query is None or measurement.value_name != "limiting_magnitude"
    ):
        if measurement.pipeline == "night":
            return Placeholder("No limiting-magnitude map", "The night pipeline gave no limiting magnitudes.")
        return _waiting_for_dark(schedule)
    if kind == "extinction" and measurement.extinction_query is None:
        if measurement.pipeline == "night":
            return Placeholder(
                "No extinction map", "It needs linear (raw) images: the night pipeline's photometric mode."
            )
        return _waiting_for_dark(schedule)
    return None
