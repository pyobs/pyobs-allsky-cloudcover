from __future__ import annotations

from typing import Dict, Any

from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_options import CatalogConstructorOptions
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.cloud_info_calculator_options import \
    CloudInfoCalculatorOptions
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_options import PreprocessorOptions
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import StarReverseMatcherOptions


class NightPipelineOptions(object):
    def __init__(self,
                 preprocessor_options: PreprocessorOptions,
                 catalog_options: CatalogConstructorOptions,
                 star_matcher_options: StarReverseMatcherOptions,
                 cloud_generator_options: CloudMapGeneratorOptions,
                 coverage_info_options: CloudInfoCalculatorOptions
                 ) -> None:

        self.preprocessor_options = preprocessor_options
        self.catalog_options = catalog_options
        self.star_matcher_options = star_matcher_options
        self.cloud_generator_options = cloud_generator_options
        self.coverage_info_options = coverage_info_options

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> NightPipelineOptions:
        preprocessor_options = PreprocessorOptions.from_dict(options.get("preprocessor", {}))
        catalog_options = CatalogConstructorOptions.from_dict(options.get("catalog", {}))
        star_matcher_options = StarReverseMatcherOptions.from_dict(options.get("reverse_matcher", {}))
        cloud_generator_options = CloudMapGeneratorOptions.from_dict(options.get("cloud_map", {}))
        coverage_info_options = CloudInfoCalculatorOptions.from_dict(options.get("coverage_info", {}))

        return cls(preprocessor_options,
                   catalog_options,
                   star_matcher_options,
                   cloud_generator_options,
                   coverage_info_options
                   )
