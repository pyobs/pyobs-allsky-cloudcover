from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import LimMagnitudeMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator_options import LimMagnitudeMapGeneratorOptions


class LimMagnitudeMapGeneratorFactory(object):
    def __init__(self, options: LimMagnitudeMapGeneratorOptions) -> None:
        self._options = options

    def __call__(self) -> LimMagnitudeMapGenerator:
        return LimMagnitudeMapGenerator(self._options.radius)
