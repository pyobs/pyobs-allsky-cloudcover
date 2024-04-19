from pyobs_cloudcover.pipeline.intervall import Interval


def test_upper_lower_bound():
    interval = Interval(start=0, end=10)
    assert (5 in interval) == True
    assert (-1 in interval) == False


def test_upper_bound():
    interval = Interval(start=None, end=-18)
    assert (-20 in interval) == True
    assert (11 in interval) == False


def test_lower_bound():
    interval = Interval(start=0, end=None)
    assert (-1 in interval) == False
    assert (11 in interval) == True


def test_no_bound():
    interval = Interval(start=None, end=None)
    assert (-1 in interval) == True
    assert (11 in interval) == True


def test_intersect_none():
    first = Interval(start=0, end=10)
    second = Interval(start=None, end=None)

    assert first.does_intersect(second) == True


def test_intersect_same():
    first = Interval(start=0, end=10)
    second = Interval(start=0, end=10)

    assert first.does_intersect(second) == True


def test_intersect_superset():
    first = Interval(start=-10, end=20)
    second = Interval(start=0, end=10)

    assert first.does_intersect(second) == True


def test_intersect_half_open():
    first = Interval(start=None, end=5)
    second = Interval(start=0, end=None)

    assert first.does_intersect(second) == True


def test_equal_different():
    first = Interval(start=None, end=5)
    second = Interval(start=0)

    assert (first == second) == False


def test_equal_same():
    first = Interval(start=None)
    second = Interval(start=None)

    assert (first == second) == True


def test_equal_invalid():
    first = Interval(start=None)
    second = object()

    assert (first == second) == False
