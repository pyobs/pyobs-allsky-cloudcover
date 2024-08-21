from __future__ import annotations

from typing import Dict, Any, cast

from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions


class DayPipelineOptions:
    def __init__(self, model_options: Dict[str, Any], mask_filepath: str, sun_apparent_size: float, cloud_map: CloudMapGeneratorOptions) -> None:
        self.model_options = model_options
        self.cloud_map = cloud_map
        self.mask_filepath = mask_filepath
        self.sun_apparent_size = sun_apparent_size

    @classmethod
    def from_dict(cls, options: Dict[str, Dict[str, Any]]) -> DayPipelineOptions:
        model_options = options.get("world_model", {})
        mask_filepath = cast(str, options.get('mask_filepath'))
        sun_apparent_size = float(cast(float, options.get('sun_apparent_size')))
        cloud_map = CloudMapGeneratorOptions.from_dict(cast(Dict[str, Any], options.get('cloud_map')))

        return DayPipelineOptions(model_options, mask_filepath, sun_apparent_size, cloud_map)
