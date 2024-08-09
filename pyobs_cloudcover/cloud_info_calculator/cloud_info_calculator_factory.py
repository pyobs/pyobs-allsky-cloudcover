from pyobs_cloudcover.cloud_info_calculator.cloud_info_calculator_options import \
    CloudInfoCalculatorOptions
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator.zenith_cloud_coverage_calculator import ZenithCloudCoverageCalculator


class CloudInfoCalculatorFactory(object):
    def __init__(self, options: CloudInfoCalculatorOptions):
        self._options = options

    def __call__(self) -> CoverageInfoCalculator:
        coverage_change_calculator = CoverageChangeCalculator()
        zenith_masker = ZenithCloudCoverageCalculator(self._options.zenith_altitude)
        cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator, zenith_masker)

        return cloud_coverage_info_calculator
