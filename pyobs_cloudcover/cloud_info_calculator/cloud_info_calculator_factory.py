from pyobs_cloudcover.cloud_info_calculator.cloud_info_calculator_options import \
    CloudInfoCalculatorOptions
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator.zenith_cloud_coverage_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.world_model import WorldModel


class CloudInfoCalculatorFactory(object):
    def __init__(self, options: CloudInfoCalculatorOptions, model: WorldModel):
        self._options = options
        self._model = model

    def __call__(self) -> CoverageInfoCalculator:
        coverage_change_calculator = CoverageChangeCalculator()
        zenith_masker = ZenithCloudCoverageCalculator(self._options.altitude_limit)
        cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator, zenith_masker)

        return cloud_coverage_info_calculator
