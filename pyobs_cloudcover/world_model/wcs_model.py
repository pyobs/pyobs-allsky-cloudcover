import datetime
from typing import Union, Tuple

import numpy as np
from astroplan import Observer
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS
from numpy import typing as npt

from pyobs_cloudcover.world_model import WorldModel
from pyobs_cloudcover.world_model.hour_angle_converter import HourAngleConverter

from astropy import units as u


class WCSModel(WorldModel):
    def __init__(self, wcs: WCS, observer: Observer, config_obs_time: datetime.datetime) -> None:
        self._wcs = wcs
        self._observer = observer
        self._config_obs_time = config_obs_time

        self._ha_conv = HourAngleConverter(observer)

    def pix_to_altaz(self, x: Union[npt.NDArray[np.float_], float], y: Union[npt.NDArray[np.float_], float]) -> \
            Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:

        coord = self._wcs.pixel_to_world(x, y)
        ha = self._ha_conv(Angle(coord.ra.deg, unit="deg"), self._config_obs_time)

        coord = SkyCoord(ra=ha * u.rad, dec=coord.dec, location=self._observer.location, obstime=self._config_obs_time)
        coord = coord.transform_to("altaz")
        return coord.alt.rad, coord.az.rad

    def altaz_to_pix(self, alt: Union[npt.NDArray[np.float_], float], az: Union[npt.NDArray[np.float_], float]) -> \
                Union[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], Tuple[float, float]]:

        coord = SkyCoord(alt=alt * u.rad, az=az * u.rad, frame="altaz",
                         location=self._observer.location, obstime=self._config_obs_time)
        coord = coord.transform_to(frame="icrs")
        ha = self._ha_conv(Angle(coord.ra.deg, unit="deg"), self._config_obs_time)

        coord = SkyCoord(ra=ha * u.rad, dec=coord.dec, location=self._observer.location, obstime=self._config_obs_time)

        pixel = self._wcs.world_to_pixel(coord)
        return pixel[0], pixel[1]
