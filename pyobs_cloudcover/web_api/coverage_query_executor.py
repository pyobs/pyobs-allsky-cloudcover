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

    def get_obs_time(self) -> Optional[float]:
        if self._cloud_query_info is None:
            return None

        obs_time: datetime.datetime = self._cloud_query_info[1]
        return obs_time.timestamp()

    def point_query_radec(self, ra: float, dec: float) -> Optional[bool]:
        if self._cloud_query_info is None:
            return None

        cloud_query, obs_time = self._cloud_query_info

        alt, az = self._radec_to_altaz(ra, dec, obs_time)

        return self.point_query_altaz(alt, az)

    def _radec_to_altaz(self, ra: float, dec: float, obs_time: datetime.datetime) -> Tuple[float, float]:
        coord = SkyCoord(ra, dec, unit='deg', frame="icrs", location=self._observer.location, obstime=obs_time)
        coord = coord.altaz

        return coord.alt.deg, coord.az.deg

    def point_query_altaz(self, alt: float, az: float) -> Optional[bool]:
        if self._cloud_query_info is None:
            return None

        cloud_query, obs_time = self._cloud_query_info

        cloudiness = cloud_query.query_nearest_coordinate(AltAzCoord(np.deg2rad(alt), np.deg2rad(az)))

        if cloudiness is None:
            return None
        else:
            return bool(cloudiness)

    def area_query_radec(self, ra: float, dec: float, radius: float) -> Optional[float]:
        if self._cloud_query_info is None:
            return None
        cloud_query, obs_time = self._cloud_query_info

        alt, az = self._radec_to_altaz(ra, dec, obs_time)

        return self.area_query_altaz(alt, az, radius)

    def area_query_altaz(self, alt: float, az: float, radius: float) -> Optional[float]:
        if self._cloud_query_info is None:
            return None
        cloud_query, obs_time = self._cloud_query_info

        cloudiness = cloud_query.query_radius(AltAzCoord(np.deg2rad(alt), np.deg2rad(az)), np.deg2rad(radius))

        if cloudiness is None:
            return None
        else:
            cloud_fraction = 100.0 * float(cloudiness)
            return cloud_fraction
