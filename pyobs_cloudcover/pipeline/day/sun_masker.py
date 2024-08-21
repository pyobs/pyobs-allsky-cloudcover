import datetime

import numpy as np
from cloudmap_rs import SkyPixelQuery, AltAzCoord

from astroplan import Observer


class SunMasker:
    def __init__(self, sun_apparent_size: float, observer: Observer):
        self._sun_apparent_size = sun_apparent_size
        self._observer = observer

    def __call__(self, sky_query: SkyPixelQuery, obs_time: datetime.datetime) -> SkyPixelQuery:
        sun_alt_az = self._observer.sun_altaz(obs_time)
        sky_query.mask_radius(AltAzCoord(sun_alt_az.alt.rad, sun_alt_az.az.rad), np.deg2rad(self._sun_apparent_size))

        return sky_query
