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
        self._measurement: CloudCoverageInfo | None = None

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._cloud_query_info = (measurement.cloud_cover_query, measurement.obs_time)
        self._measurement = measurement

    @property
    def measurement(self) -> CloudCoverageInfo | None:
        return self._measurement

    def altaz(self, ra: float | None, dec: float | None, alt: float | None, az: float | None) -> tuple[float, float]:
        """Position (deg) from either RA/Dec (converted for the latest observation time) or alt/az."""
        if ra is not None and dec is not None:
            if self._measurement is None:
                raise ValueError("No measurement yet")
            return self._radec_to_altaz(ra, dec, self._measurement.obs_time)
        if alt is None or az is None:
            raise ValueError("Need ra/dec or alt/az")
        return alt, az

    def limiting_magnitude(self, alt: float, az: float) -> float | None:
        """Limiting magnitude at the analysed position closest to (alt, az) in degrees, if the night pipeline ran."""
        m = self._measurement
        if m is None or m.value_query is None or m.value_name != "limiting_magnitude":
            return None
        return m.value_query.query_nearest_coordinate(np.deg2rad(alt), np.deg2rad(az))

    def limiting_magnitude_area(self, alt: float, az: float, radius: float) -> dict[str, float] | None:
        """Mean, minimum and maximum limiting magnitude within ``radius`` degrees."""
        m = self._measurement
        if m is None or m.value_query is None or m.value_name != "limiting_magnitude":
            return None
        q = m.value_query
        values = q.get_pixels()[q.grid.within(np.deg2rad(alt), np.deg2rad(az), np.deg2rad(radius))]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None
        return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}

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
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs", location=self._observer.location, obstime=obs_time)
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
