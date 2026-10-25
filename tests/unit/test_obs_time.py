import datetime

import pytest

from pyobs_cloudcover.obs_time import parse_time, time_from_header

UTC = datetime.UTC


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2024-03-09T03:05:09.660376", datetime.datetime(2024, 3, 9, 3, 5, 9, 660376, UTC)),
        # regression: whole seconds used to raise (format required fractional seconds)
        ("2024-03-09T03:05:09", datetime.datetime(2024, 3, 9, 3, 5, 9, tzinfo=UTC)),
        ("2024-03-09T03:05:09.123", datetime.datetime(2024, 3, 9, 3, 5, 9, 123000, UTC)),
        ("2024-03-09T03:05:09.1234567", datetime.datetime(2024, 3, 9, 3, 5, 9, 123457, UTC)),
        ("2024-03-09 03:05", datetime.datetime(2024, 3, 9, 3, 5, tzinfo=UTC)),
        ("2024-03-09T03:05:09Z", datetime.datetime(2024, 3, 9, 3, 5, 9, tzinfo=UTC)),
        ("2024-03-09T05:05:09+02:00", datetime.datetime(2024, 3, 9, 3, 5, 9, tzinfo=UTC)),
    ],
)
def test_parse_time(value: str, expected: datetime.datetime) -> None:
    assert parse_time(value) == expected


def test_local_timezone() -> None:
    assert parse_time("2024-03-09T05:05:09", "Africa/Johannesburg") == datetime.datetime(
        2024, 3, 9, 3, 5, 9, tzinfo=UTC
    )


def test_header_date_and_time() -> None:
    assert time_from_header({"DATE-OBS": "2024-03-09", "TIME-OBS": "03:05:09.5"}) == datetime.datetime(
        2024, 3, 9, 3, 5, 9, 500000, UTC
    )
    assert time_from_header({"DATE": "2024-03-09"}) is None
    assert time_from_header({"DATE_OBS": "2024-03-09T03:05:09"}) is not None


def test_garbage() -> None:
    with pytest.raises(ValueError):
        parse_time("yesterday")
