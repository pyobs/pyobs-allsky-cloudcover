from astroplan import Observer

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_options import CatalogConstructorOptions
from pyobs_cloudcover.world_model import WorldModel


class CatalogConstructorFactory(object):
    def __init__(self, options: CatalogConstructorOptions, model: WorldModel, observer: Observer) -> None:
        self._options = options
        self._model = model
        self._observer = observer

    def __call__(self) -> CatalogConstructor:
        altaz_catalog_loader = AltAzCatalogLoader.from_csv(self._options.filepath)
        catalog_constructor = CatalogConstructor(
            altaz_catalog_loader, self._model, self._observer,
            self._options.alt_filter, self._options.v_mag_filter, self._options.distance_filter
        )

        return catalog_constructor
