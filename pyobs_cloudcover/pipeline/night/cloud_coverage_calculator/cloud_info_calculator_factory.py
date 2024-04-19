from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.cloud_info_calculator_options import \
    CloudInfoCalculatorOptions
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.zenith_masker import ZenithMasker
from pyobs_cloudcover.world_model import WorldModel


class CloudInfoCalculatorFactory(object):
    def __init__(self, options: CloudInfoCalculatorOptions, model: WorldModel):
        self._options = options
        self._model = model

    def __call__(self) -> CoverageInfoCalculator:
        coverage_calculator = CoverageCalculator(self._options.cloud_threshold)
        coverage_change_calculator = CoverageChangeCalculator()
        zenith_masker = ZenithMasker(self._options.altitude_limit, self._model)
        cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator,
                                                                zenith_masker)

        return cloud_coverage_info_calculator
