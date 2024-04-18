import datetime
from typing import Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from astropy.coordinates import SkyCoord

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.pipeline.night.world_model.world_model import WorldModel


class CoverageQueryExecutor(object):
    def __init__(self, model: WorldModel, observer: Observer, window: ImageWindow) -> None:
        self._model = model
        self._observer = observer
        self._window = window

        self._cloud_map: Optional[Tuple[npt.NDArray[np.float_], datetime.datetime]] = None

    def set_measurement(self, measurement: CloudCoverageInfo) -> None:
        self._cloud_map = (measurement.cloud_cover_map, measurement.obs_time)

    def get_obs_time(self) -> float:
        obs_time: datetime.datetime = self._cloud_map[1]
        return obs_time.timestamp()

    def __call__(self, ra: float, dec: float) -> Optional[float]:
        if self._cloud_map is None:
            return None

        cloud_map, _ = self._cloud_map
        self._window.set_image(cloud_map)

        alt, az = self._radec_to_altaz(ra, dec)
        px, py = self._model.altaz_to_pix(alt, az)
        cloud_area = self._window(cast(float, px), cast(float, py))

        average_cover = np.average(cloud_area[~np.isnan(cloud_area)])

        if np.isnan(average_cover):
            return None
        else:
            return float(average_cover)

    def _radec_to_altaz(self, ra: float, dec: float) -> Tuple[float, float]:
        _, obs_time = self._cloud_map

        coord = SkyCoord(ra, dec, unit='deg', frame="ircs", location=self._observer.location, obstime=obs_time)
        coord = coord.altaz

        return coord.alt.rad, coord.az.rad
