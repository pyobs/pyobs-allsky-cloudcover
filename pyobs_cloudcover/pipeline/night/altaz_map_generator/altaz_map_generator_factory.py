from pyobs_cloudcover.pipeline.night.altaz_map_generator.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_options import CatalogConstructorOptions
from pyobs_cloudcover.world_model import WorldModel


class AltAzMapGeneratorFactory(object):
    def __init__(self, options: CatalogConstructorOptions, model: WorldModel):
        self._limiting_altitude = options.alt_filter
        self._model = model

    def __call__(self) -> AltAzMapGenerator:
        return AltAzMapGenerator(self._model, self._limiting_altitude)
