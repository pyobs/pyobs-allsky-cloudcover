from astroplan import Observer

from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_factory import CatalogConstructorFactory
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.cloud_info_calculator_factory import \
    CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_factory import PreprocessorFactory
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_factory import StarReverseMatcherFactory
from pyobs_cloudcover.world_model import WorldModel


class NightPipelineFactory(object):
    def __init__(self, observer: Observer, model: WorldModel):
        self._observer = observer
        self._model = model

    def __call__(self, options: NightPipelineOptions) -> NightPipeline:
        preprocessor_factory = PreprocessorFactory(options.preprocessor_options)
        catalog_constructor_factory = CatalogConstructorFactory(options.catalog_options, self._model, self._observer)
        reverse_matcher_factory = StarReverseMatcherFactory(options.star_matcher_options)
        cloud_map_generator_factory = CloudMapGeneratorFactory(options.cloud_generator_options)
        coverage_info_calculator_factory = CloudInfoCalculatorFactory(options.coverage_info_options, self._model)

        preprocessor = preprocessor_factory()
        catalog_constructor = catalog_constructor_factory()
        star_reverse_matcher = reverse_matcher_factory()
        cloud_map_generator = cloud_map_generator_factory()
        coverage_info_calculator = coverage_info_calculator_factory()

        pipeline = NightPipeline(
            preprocessor,
            catalog_constructor,
            star_reverse_matcher,
            cloud_map_generator,
            coverage_info_calculator
        )

        return pipeline