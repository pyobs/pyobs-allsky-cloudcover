from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_info_calculator import CoverageInfoCalculator


class CloudInfoCalculatorFactory(object):

    def __call__(self) -> CoverageInfoCalculator:
        coverage_change_calculator = CoverageChangeCalculator()
        cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator)

        return cloud_coverage_info_calculator
