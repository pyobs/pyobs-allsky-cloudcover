from __future__ import annotations

from typing import Dict, Any

from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_grid_options import AltAzGridOptions
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_options import CatalogConstructorOptions
from pyobs_cloudcover.cloud_info_calculator.cloud_info_calculator_options import \
    CloudInfoCalculatorOptions
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator_options import LimMagnitudeMapGeneratorOptions
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_options import PreprocessorOptions
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import StarReverseMatcherOptions


class NightPipelineOptions(object):
    def __init__(self,
                 model_options: Dict[str, Any],
                 preprocessor_options: PreprocessorOptions,
                 catalog_options: CatalogConstructorOptions,
                 star_matcher_options: StarReverseMatcherOptions,
                 lim_mag_map_generator_options: LimMagnitudeMapGeneratorOptions,
                 cloud_generator_options: CloudMapGeneratorOptions,
                 coverage_info_options: CloudInfoCalculatorOptions,
                 altaz_grid_options: AltAzGridOptions
                 ) -> None:

        self.model_options = model_options
        self.altaz_grid_options = altaz_grid_options
        self.preprocessor_options = preprocessor_options
        self.catalog_options = catalog_options
        self.star_matcher_options = star_matcher_options
        self.lim_mag_map_generator_options = lim_mag_map_generator_options
        self.cloud_generator_options = cloud_generator_options
        self.coverage_info_options = coverage_info_options

    @classmethod
    def from_dict(cls, options: Dict[str, Dict[str, Any]]) -> NightPipelineOptions:
        model_options = options.get("world_model", {})
        preprocessor_options = PreprocessorOptions.from_dict(options.get("preprocessor", {}))
        catalog_options = CatalogConstructorOptions.from_dict(options.get("catalog", {}))
        star_matcher_options = StarReverseMatcherOptions.from_dict(options.get("reverse_matcher", {}))
        lim_mag_map_generator_options = LimMagnitudeMapGeneratorOptions.from_dict(options.get("lim_mag_map", {}))
        cloud_map_generator_options = CloudMapGeneratorOptions.from_dict(options.get("cloud_map", {}))
        coverage_info_options = CloudInfoCalculatorOptions.from_dict(options.get("coverage_info", {}))
        altaz_grid_generator = AltAzGridOptions.from_dict(options.get("altaz_grid", {}))

        return cls(model_options,
                   preprocessor_options,
                   catalog_options,
                   star_matcher_options,
                   lim_mag_map_generator_options,
                   cloud_map_generator_options,
                   coverage_info_options,
                   altaz_grid_generator
                   )
