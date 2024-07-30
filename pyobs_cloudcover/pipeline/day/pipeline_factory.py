import numpy as np

from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.day.pipeline import DayPipeline
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.world_model import WorldModel


class DayPipelineFactory(object):
    def __init__(self, model: WorldModel):
        self._model = model

    def __call__(self, options: DayPipelineOptions) -> DayPipeline:
        mask = np.load(options.mask_filepath)
        alt_az_generator = AltAzMapGenerator(self._model, 0)
        cloud_map_generator_factory = CloudMapGeneratorFactory(options.cloud_map)
        coverage_info_calculator_factory = CloudInfoCalculatorFactory(options.coverage_info)

        return DayPipeline(mask, alt_az_generator, cloud_map_generator_factory(), coverage_info_calculator_factory())
