"""The pipelines' sun-altitude windows, in words."""

from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.schedule import AnalysisSchedule, describe_interval


def test_describe() -> None:
    assert describe_interval(Interval(None, -18)) == "sun below -18\N{DEGREE SIGN}"
    assert describe_interval(Interval(-6, None)) == "sun above -6\N{DEGREE SIGN}"
    assert describe_interval(Interval(-12, -6)) == "sun between -12\N{DEGREE SIGN} and -6\N{DEGREE SIGN}"


def test_windows() -> None:
    s = AnalysisSchedule({"night": Interval(None, -18)})
    assert s.names == ["night"] and s.window("night") == Interval(None, -18) and s.window("day") is None
