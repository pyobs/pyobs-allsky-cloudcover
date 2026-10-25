"""Parsing of observation timestamps from FITS headers."""

from __future__ import annotations

import datetime
import re
import zoneinfo
from collections.abc import Mapping
from typing import Any

_ISO = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?\s*(Z|[+-]\d{2}:?\d{2})?\s*$"
)
_TIME = re.compile(r"^\s*(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?\s*$")


def _tz(name: str | datetime.tzinfo | None) -> datetime.tzinfo:
    if name is None:
        return datetime.UTC
    if isinstance(name, datetime.tzinfo):
        return name
    return datetime.UTC if name.upper() == "UTC" else zoneinfo.ZoneInfo(name)


def _micro(fraction: str | None) -> int:
    # any number of digits (N.I.N.A. writes 7), rounded to microseconds
    return 0 if not fraction else min(999999, round(int(fraction) / 10 ** (len(fraction) - 6)))


def parse_time(value: str, timezone: str | datetime.tzinfo | None = None, time: str | None = None) -> datetime.datetime:
    """Parse an ISO-8601-like timestamp into an aware UTC datetime.

    Accepts a ``T`` or space separator, missing seconds, any number of fractional digits, a ``Z`` or
    ``+hh:mm`` suffix, or a date only (then ``time`` must give the time of day, as in the old FITS
    ``DATE-OBS`` + ``TIME-OBS`` convention). Timestamps without a zone are taken to be in ``timezone``
    (default UTC).
    """
    m = _ISO.match(value)
    if m is None:
        raise ValueError(f"Can't parse timestamp {value!r}")
    year, month, day, hh, mm, ss, frac, zone = m.groups()
    if hh is None:
        if time is None:
            raise ValueError(f"Timestamp {value!r} has no time of day")
        t = _TIME.match(time)
        if t is None:
            raise ValueError(f"Can't parse time of day {time!r}")
        hh, mm, ss, frac = t.groups()
    if zone is None:
        tz = _tz(timezone)
    elif zone == "Z":
        tz = datetime.UTC
    else:
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        tz = datetime.timezone(sign * datetime.timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
    local = datetime.datetime(
        int(year), int(month), int(day), int(hh), int(mm), int(ss or 0), _micro(frac), tzinfo=tz
    )
    return local.astimezone(datetime.UTC)


def time_from_header(
    header: Mapping[str, Any], keywords: tuple[str, ...] = ("DATE-OBS", "DATE_OBS"), timezone: str | None = None
) -> datetime.datetime | None:
    """Observation time from the first of ``keywords`` present in the header (with ``TIME-OBS`` if the keyword
    only holds a date), as aware UTC; None if no keyword is present."""
    for key in keywords:
        if key in header and str(header[key]).strip():
            time_of_day = header.get("TIME-OBS")
            return parse_time(str(header[key]), timezone, str(time_of_day) if time_of_day is not None else None)
    return None
