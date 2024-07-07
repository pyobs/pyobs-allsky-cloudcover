from __future__ import annotations

import datetime

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from astropy.coordinates import SkyCoord

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog import AltAzCatalog


class AltAzCatalogLoader(object):
    def __init__(self, sao: npt.NDArray[np.int_], ra: npt.NDArray[np.float_], dec: npt.NDArray[np.float_], v_mag: npt.NDArray[np.float_]) -> None:
        self._sao = sao
        self._ra = ra
        self._dec = dec
        self._vmag = v_mag

    def __call__(self, observer: Observer, obstime: datetime.datetime) -> AltAzCatalog:
        alt, az = self._equatorial_to_altaz(observer, obstime)

        return AltAzCatalog(self._sao, alt, az, self._vmag)

    def _equatorial_to_altaz(self, observer: Observer, obstime: datetime.datetime) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ircs_coords = SkyCoord(ra=self._ra, dec=self._dec, unit='deg', location=observer.location, obstime=obstime)
        alt_az_coord = ircs_coords.transform_to("altaz")
        return alt_az_coord.alt.deg, alt_az_coord.az.deg

    @classmethod
    def from_csv(cls, file: str) -> AltAzCatalogLoader:
        catalog_file = np.loadtxt(file, skiprows=1, delimiter=";", comments=["#"])

        return cls(catalog_file[:, 1].astype(np.int_), catalog_file[:, 2], catalog_file[:, 3], catalog_file[:, 4])
