import datetime

from astroplan import Observer
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from pyobs_cloudcover.world_model.wcs_model import WCSModel


class WCSModelLoader(object):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def __call__(self) -> WCSModel:
        wcs_image = fits.open(self._file_path)[0]

        wcs = WCS(header=wcs_image.header)
        config_obs_time = datetime.datetime.fromisoformat(wcs_image.header["DATE-OBS"])

        observer = Observer(
            latitude=wcs_image.header["LATITUDE"] * u.deg,
            longitude=wcs_image.header["LONGITUD"] * u.deg,
            elevation=wcs_image.header["HEIGHT"] * u.m,
        )

        return WCSModel(wcs=wcs, observer=observer, config_obs_time=config_obs_time)

