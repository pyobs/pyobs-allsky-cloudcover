from pyobs_cloudcover.pipeline.intervall import Interval


def test_upper_lower_bound():
    interval = Interval(start=0, end=10)
    assert (5 in interval)
    assert -1 not in interval


def test_upper_bound():
    interval = Interval(start=None, end=-18)
    assert (-20 in interval)
    assert 11 not in interval


def test_lower_bound():
    interval = Interval(start=0, end=None)
    assert -1 not in interval
    assert (11 in interval)


def test_no_bound():
    interval = Interval(start=None, end=None)
    assert (-1 in interval)
    assert (11 in interval)


def test_intersect_none():
    first = Interval(start=0, end=10)
    second = Interval(start=None, end=None)

    assert first.does_intersect(second)


def test_intersect_same():
    first = Interval(start=0, end=10)
    second = Interval(start=0, end=10)

    assert first.does_intersect(second)


def test_intersect_superset():
    first = Interval(start=-10, end=20)
    second = Interval(start=0, end=10)

    assert first.does_intersect(second)


def test_intersect_half_open():
    first = Interval(start=None, end=5)
    second = Interval(start=0, end=None)

    assert first.does_intersect(second)


def test_equal_different():
    first = Interval(start=None, end=5)
    second = Interval(start=0)

    assert not (first == second)


def test_equal_same():
    first = Interval(start=None)
    second = Interval(start=None)

    assert (first == second)


def test_equal_invalid():
    first = Interval(start=None)
    second = object()

    assert not (first == second)


def test_boundary_belongs_to_exactly_one_interval() -> None:
    # regression: with strict comparisons on both ends, a sun altitude of exactly -18 matched no pipeline
    night = Interval(start=None, end=-18)
    twilight = Interval(start=-18, end=18)
    assert -18 not in night
    assert -18 in twilight
    assert not night.does_intersect(twilight)


def test_overlap_detected() -> None:
    assert Interval(start=None, end=-17).does_intersect(Interval(start=-18, end=None))
