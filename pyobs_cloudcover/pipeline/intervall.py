from __future__ import annotations
from typing import Optional


class Interval(object):
    def __init__(self, start: Optional[float] = None, end: Optional[float] = None):
        self._start = start
        self._end = end

    def __contains__(self, value: float) -> bool:
        in_interval = True

        if self._start is not None:
            in_interval &= self._start < value

        if self._end is not None:
            in_interval &= value < self._end

        return in_interval

    def does_intersect(self, other: Interval) -> bool:
        if (other._start is None and other._end is None) or (self._start is None and self._end is None):
            return True

        if self == other:
            return True

        does_intersect = False

        if other._start is not None:
            does_intersect |= other._start in self

        if other._end is not None:
            does_intersect |= other._end in self

        if self._start is not None:
            does_intersect |= self._start in other

        if self._end is not None:
            does_intersect |= self._end in other

        return does_intersect

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return False

        return self._start == other._start and self._end == other._end
