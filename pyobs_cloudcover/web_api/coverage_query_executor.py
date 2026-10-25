import datetime

import numpy as np
from astroplan import Observer
from astropy.coordinates import SkyCoord

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.sky_query import SkyPixelQuery


class CoverageQueryExecutor:
    def __init__(self, observer: Observer) -> None:
        self._observer = observer

        self._cloud_query_info: tuple[SkyPixelQuery, datetime.datetime] | None = None

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._cloud_query_info = (measurement.cloud_cover_query, measurement.obs_time)

    def get_obs_time(self) -> float | None:
        if self._cloud_query_info is None:
            return None

        obs_time: datetime.datetime = self._cloud_query_info[1]
        return obs_time.timestamp()

    def point_query_radec(self, ra: float, dec: float) -> bool | None:
        if self._cloud_query_info is None:
            return None

        cloud_query, obs_time = self._cloud_query_info

        alt, az = self._radec_to_altaz(ra, dec, obs_time)

        return self.point_query_altaz(alt, az)

    def _radec_to_altaz(self, ra: float, dec: float, obs_time: datetime.datetime) -> tuple[float, float]:
        coord = SkyCoord(ra, dec, unit='deg', frame="icrs", location=self._observer.location, obstime=obs_time)
        coord = coord.altaz

        return coord.alt.deg, coord.az.deg

    def point_query_altaz(self, alt: float, az: float) -> bool | None:
        if self._cloud_query_info is None:
            return None

        cloud_query, obs_time = self._cloud_query_info

        cloudiness = cloud_query.query_nearest_coordinate(np.deg2rad(alt), np.deg2rad(az))

        if cloudiness is None:
            return None
        else:
            return bool(cloudiness)

    def area_query_radec(self, ra: float, dec: float, radius: float) -> float | None:
        if self._cloud_query_info is None:
            return None
        cloud_query, obs_time = self._cloud_query_info

        alt, az = self._radec_to_altaz(ra, dec, obs_time)

        return self.area_query_altaz(alt, az, radius)

    def area_query_altaz(self, alt: float, az: float, radius: float) -> float | None:
        """Cloud fraction (0..1) within ``radius`` degrees."""
        if self._cloud_query_info is None:
            return None
        cloud_query, obs_time = self._cloud_query_info

        cloudiness = cloud_query.query_radius(np.deg2rad(alt), np.deg2rad(az), np.deg2rad(radius))

        return cloudiness
