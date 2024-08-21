from astroplan import Observer
from pyobs.object import get_object

from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_map_generator_factory import AltAzMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_factory import CatalogConstructorFactory
from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator_factory import LimMagnitudeMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_factory import PreprocessorFactory
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_factory import StarReverseMatcherFactory
from pyobs_cloudcover.world_model import WorldModel
from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory


class NightPipelineFactory(object):
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, options: NightPipelineOptions) -> NightPipeline:
        world_model_factory = WorldModelFactory(self._observer)
        model: WorldModel = world_model_factory(options.model_options)

        preprocessor_factory = PreprocessorFactory(options.preprocessor_options)
        catalog_constructor_factory = CatalogConstructorFactory(options.catalog_options, model, self._observer)
        altaz_map_generator_factory = AltAzMapGeneratorFactory(options.altaz_grid_options)
        reverse_matcher_factory = StarReverseMatcherFactory(options.star_matcher_options)
        lim_mag_map_generator_factory = LimMagnitudeMapGeneratorFactory(options.lim_mag_map_generator_options)
        cloud_map_generator_factory = CloudMapGeneratorFactory(options.cloud_generator_options)
        coverage_info_calculator_factory = CloudInfoCalculatorFactory()

        preprocessor = preprocessor_factory()
        catalog_constructor = catalog_constructor_factory()
        altaz_map_generator = altaz_map_generator_factory()
        star_reverse_matcher = reverse_matcher_factory()
        lin_mag_map_generator = lim_mag_map_generator_factory()
        cloud_map_generator = cloud_map_generator_factory()
        coverage_info_calculator = coverage_info_calculator_factory()
        moon_masker = MoonMasker(options.moon_apparent_size, self._observer)

        pipeline = NightPipeline(
            preprocessor,
            catalog_constructor,
            altaz_map_generator,
            star_reverse_matcher,
            lin_mag_map_generator,
            cloud_map_generator,
            moon_masker,
            coverage_info_calculator
        )

        return pipeline
