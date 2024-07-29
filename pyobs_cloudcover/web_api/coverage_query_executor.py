import datetime
from typing import Optional, Tuple

import numpy as np
from astroplan import Observer
from astropy.coordinates import SkyCoord
from cloudmap_rs import AltAzCoord, SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo


class CoverageQueryExecutor(object):
    def __init__(self, observer: Observer) -> None:
        self._observer = observer

        self._cloud_query_info: Optional[Tuple[SkyPixelQuery, datetime.datetime]] = None

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._cloud_query_info = (measurement.cloud_cover_query, measurement.obs_time)

    def get_obs_time(self) -> float:
        if self._cloud_query_info is None:
            raise ValueError("Measurement has not been set yet!")

        obs_time: datetime.datetime = self._cloud_query_info[1]
        return obs_time.timestamp()

    def __call__(self, ra: float, dec: float) -> Optional[bool]:
        if self._cloud_query_info is None:
            return None

        cloud_query, obs_time = self._cloud_query_info

        alt, az = self._radec_to_altaz(ra, dec, obs_time)

        cloudiness = cloud_query.query_nearest_coordinate(AltAzCoord(alt, az))

        if cloudiness is None:
            return None
        else:
            return bool(cloudiness)

    def _radec_to_altaz(self, ra: float, dec: float, obs_time: datetime.datetime) -> Tuple[float, float]:
        coord = SkyCoord(ra, dec, unit='deg', frame="ircs", location=self._observer.location, obstime=obs_time)
        coord = coord.altaz

        return coord.alt.rad, coord.az.rad
