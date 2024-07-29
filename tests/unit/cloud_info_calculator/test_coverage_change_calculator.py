from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator


def test_fist():
    cloud_map = [True, False, True, False]
    calculator = CoverageChangeCalculator()
    assert calculator(cloud_map) is None


def test_call():
    calculator = CoverageChangeCalculator()
    calculator._last_map = [True, False, True, False]

    cloud_map = [None, True, True, True]

    assert calculator(cloud_map) == 2/3
