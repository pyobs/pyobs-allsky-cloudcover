from __future__ import annotations

import math


class Interval:
    """Half-open interval [start, end); None means unbounded."""

    def __init__(self, start: float | None = None, end: float | None = None):
        self._start = start
        self._end = end

    @property
    def start(self) -> float | None:
        return self._start

    @property
    def end(self) -> float | None:
        return self._end

    @property
    def _lower(self) -> float:
        return -math.inf if self._start is None else self._start

    @property
    def _upper(self) -> float:
        return math.inf if self._end is None else self._end

    def __contains__(self, value: float) -> bool:
        return self._lower <= value < self._upper

    def does_intersect(self, other: Interval) -> bool:
        return max(self._lower, other._lower) < min(self._upper, other._upper)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return False

        return self._start == other._start and self._end == other._end
