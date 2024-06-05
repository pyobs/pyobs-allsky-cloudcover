import datetime

from astroplan import Observer

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.world_model import WorldModel


class CatalogConstructor(object):
    def __init__(self, catalog_loader: AltAzCatalogLoader, model: WorldModel, observer: Observer,
                 alt_limit: float, v_mag_limit: float, distance_limit: float) -> None:
        self._catalog_loader = catalog_loader
        self._model = model
        self._observer = observer
        self._alt_limit = alt_limit
        self._v_mag_limit = v_mag_limit
        self._distance_limit = distance_limit

    def __call__(self, obs_time: datetime.datetime, height: int, width: int) -> PixelCatalog:
        altaz_catalog = self._catalog_loader(self._observer, obs_time)
        altaz_catalog.filter_alt(self._alt_limit)
        altaz_catalog.filter_v_mag(self._v_mag_limit)

        pixel_catalog = PixelCatalog.from_altaz_catalog(altaz_catalog, self._model)
        pixel_catalog.filter_window_size(height, width)
        pixel_catalog.filter_close(self._distance_limit)

        return pixel_catalog
