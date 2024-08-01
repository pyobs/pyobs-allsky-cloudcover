from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions


class CloudMapGeneratorFactory:
    def __init__(self, options: CloudMapGeneratorOptions):
        self._options = options

    def __call__(self) -> CloudMapGenerator:
<<<<<<< HEAD
        return CloudMapGenerator(self._options.threshold)
=======
        return CloudMapGenerator(self._options.radius, self._options.integration_length)
>>>>>>> main
