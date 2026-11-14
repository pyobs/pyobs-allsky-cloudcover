"""Why an image route shows a placeholder instead of a real image."""

import datetime
import io

import numpy as np
import pytest
from PIL import Image

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.schedule import AnalysisSchedule
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery
from pyobs_cloudcover.web_api.placeholders import Placeholder, placeholder_reason
from pyobs_cloudcover.web_api.render import render_placeholder

NOW = datetime.datetime(2026, 11, 13, 12, 0, tzinfo=datetime.UTC)
MIN = datetime.timedelta(minutes=1)


@pytest.fixture(scope="module")
def schedule() -> AnalysisSchedule:
    return AnalysisSchedule({"night": Interval(None, -18)})


def _result(time: datetime.datetime, pipeline: str = "night") -> CloudCoverageInfo:
    q = SkyPixelQuery(SkyGrid([1.0], [0.0]), [0.0])
    if pipeline == "night":
        return CloudCoverageInfo(q, None, time, q, "limiting_magnitude", "night")
    return CloudCoverageInfo(q, None, time, q, "color_ratio", "day")


def reason(kind, **kw):  # type: ignore[no-untyped-def]
    kw.setdefault("stale_after", 900)
    return placeholder_reason(kind, NOW, **kw)


def test_real_images_when_everything_is_fresh() -> None:
    fresh = dict(frame_time=NOW - 2 * MIN, measurement=_result(NOW - 2 * MIN))
    for kind in ("image", "sky", "cloud_map", "limiting_magnitude"):
        assert reason(kind, **fresh) is None


def test_nothing_yet() -> None:
    assert reason("image", frame_time=None, measurement=None).title == "No image yet"
    assert reason("cloud_map", frame_time=None, measurement=None).title == "No image yet"


def test_camera_stopped() -> None:
    r = reason("cloud_map", frame_time=NOW - 30 * MIN, measurement=_result(NOW - 30 * MIN))
    assert r.title == "No new image" and "30 min ago" in r.detail and "2026-11-13 11:30 UT" in r.detail
    # the frame routes too: an old camera image would look current
    assert reason("image", frame_time=NOW - 30 * MIN, measurement=None).title == "No new image"


def test_analysis_failed() -> None:
    r = reason(
        "cloud_map",
        frame_time=NOW - 5 * MIN,
        measurement=_result(NOW - 5 * MIN),
        last_error="x.jpg: OSError: truncated",
        last_error_at=NOW - MIN,
    )
    assert r.title == "Analysis failed" and "truncated" in r.detail
    # an old error that a later result superseded doesn't count
    assert (
        reason(
            "cloud_map",
            frame_time=NOW - MIN,
            measurement=_result(NOW - MIN),
            last_error="old",
            last_error_at=NOW - 10 * MIN,
        )
        is None
    )


def test_no_pipeline_for_the_sun_altitude(schedule: AnalysisSchedule) -> None:
    night = _result(NOW - 10 * MIN)
    kw = dict(frame_time=NOW - MIN, measurement=night, schedule=schedule)
    day = reason("cloud_map", last_reason="no pipeline for the sun at 61.8 deg", **kw)
    assert day == Placeholder("Daytime pipeline is off", "Waiting for night: sun below -18\N{DEGREE SIGN}.")
    twilight = reason("cloud_map", last_reason="no pipeline for the sun at -12.0 deg", **kw)
    assert twilight.title == "No twilight pipeline" and "Waiting for night" in twilight.detail
    for kind in ("limiting_magnitude", "extinction"):
        r = reason(kind, last_reason="no pipeline for the sun at 61.8 deg", **kw)
        assert r == Placeholder("Waiting for astronomical dark", "Night pipeline: sun below -18\N{DEGREE SIGN}.")
    # without a schedule it still says what is going on
    assert (
        reason(
            "cloud_map", frame_time=NOW - MIN, measurement=night, last_reason="no pipeline for the sun at 61.8 deg"
        ).title
        == "Daytime pipeline is off"
    )
    # the camera image itself is still fine
    assert reason("image", last_reason="no pipeline for the sun at 61.8 deg", **kw) is None


def test_night_maps_while_the_day_pipeline_runs(schedule: AnalysisSchedule) -> None:
    day = dict(frame_time=NOW - MIN, measurement=_result(NOW - MIN, "day"), schedule=schedule)
    assert reason("limiting_magnitude", **day).title == "Waiting for astronomical dark"
    assert reason("extinction", **day).title == "Waiting for astronomical dark"
    assert reason("cloud_map", **day) is None
    # a night end other than -18 deg isn't astronomical dark
    other = AnalysisSchedule({"night": Interval(None, -12)})
    assert reason("limiting_magnitude", **{**day, "schedule": other}).title == "Waiting for night"
    # at night, a missing map is not the time of day
    night = dict(frame_time=NOW - MIN, measurement=_result(NOW - MIN))
    assert reason("extinction", **night).title == "No extinction map"


def test_sky_needs_a_world_model() -> None:
    r = reason("sky", frame_time=NOW - MIN, measurement=None, has_world_model=False)
    assert r.title == "No north-up image"


def test_placeholder_image_has_the_requested_size_and_text() -> None:
    png = render_placeholder("Daytime pipeline is off", "Waiting for night: sun below -18°.", 720, 405)
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert img.shape == (405, 720, 3)
    assert img[150:260, 150:570].max() > 200  # bright title text in the middle
