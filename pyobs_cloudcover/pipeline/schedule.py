"""When the configured pipelines run: their sun-altitude windows.

Used by the placeholder images, so that a dashboard showing "no cloud map" in daylight can say which pipeline it
is waiting for and at what sun altitude it starts.
"""

from __future__ import annotations

from pyobs_cloudcover.pipeline.intervall import Interval


def _deg(value: float) -> str:
    return f"{value:g}\N{DEGREE SIGN}"


def describe_interval(interval: Interval) -> str:
    """A sun-altitude window in words, e.g. 'sun below -18°'."""
    if interval.start is None and interval.end is None:
        return "any sun altitude"
    if interval.start is None:
        return f"sun below {_deg(interval.end)}"  # type: ignore[arg-type]
    if interval.end is None:
        return f"sun above {_deg(interval.start)}"
    return f"sun between {_deg(interval.start)} and {_deg(interval.end)}"


class AnalysisSchedule:
    """The sun-altitude window of each configured pipeline (name -> interval)."""

    def __init__(self, windows: dict[str, Interval]) -> None:
        self._windows = windows

    @property
    def names(self) -> list[str]:
        return list(self._windows)

    def window(self, name: str) -> Interval | None:
        """The sun-altitude window of a pipeline, None if it isn't configured."""
        return self._windows.get(name)
