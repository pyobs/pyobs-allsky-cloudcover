import numpy as np
from astroplan import Observer
from pyobs.object import get_object

from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.day.pipeline import DayPipeline
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.world_model import WorldModel


class DayPipelineFactory(object):
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, options: DayPipelineOptions) -> DayPipeline:
        model = get_object(options.model_options, WorldModel)
        mask = np.load(options.mask_filepath)
        alt_az_generator = AltAzMapGenerator(model, 0)
        cloud_map_generator_factory = CloudMapGeneratorFactory(options.cloud_map)

        sun_masker = SunMasker(options.sun_apparent_size, self._observer)
        coverage_info_calculator_factory = CloudInfoCalculatorFactory()

        return DayPipeline(mask, alt_az_generator, cloud_map_generator_factory(), sun_masker, coverage_info_calculator_factory())
