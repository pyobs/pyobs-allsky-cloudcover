from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator


def test_fist():
    cloud_map = [True, False, True, False]
    calculator = CoverageChangeCalculator()
    assert calculator(cloud_map) is None


def test_call():
    calculator = CoverageChangeCalculator()
    calculator._last_map = [True, False, True, False]

    cloud_map = [None, True, True, True]

    assert calculator(cloud_map) == 2/3


def test_change_is_measured_against_the_previous_frame() -> None:
    # regression: the map was stored in the wrong attribute, so every frame was compared to the first one
    calculator = CoverageChangeCalculator()
    calculator([0.0, 0.0, 0.0, 0.0])
    assert calculator([1.0, 1.0, 1.0, 1.0]) == 1.0
    assert calculator([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_no_overlap_gives_none() -> None:
    calculator = CoverageChangeCalculator()
    calculator([float("nan"), 1.0])
    assert calculator([1.0, float("nan")]) is None
