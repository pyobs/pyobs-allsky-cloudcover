from __future__ import annotations

from typing import Dict, Any, cast

from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorOptions
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions


class DayPipelineOptions:
    def __init__(self, mask_filepath: str, cloud_map: CloudMapGeneratorOptions, coverage_info: CloudInfoCalculatorOptions) -> None:
        self.coverage_info = coverage_info
        self.cloud_map = cloud_map
        self.mask_filepath = mask_filepath

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> DayPipelineOptions:
        mask_filepath = cast(str, options.get('mask_filepath'))
        cloud_map = CloudMapGeneratorOptions.from_dict(cast(Dict[str, Any], options.get('cloud_map')))
        coverage_info = CloudInfoCalculatorOptions.from_dict(cast(Dict[str, Any], options.get('coverage_info')))

        return DayPipelineOptions(mask_filepath, cloud_map, coverage_info)
