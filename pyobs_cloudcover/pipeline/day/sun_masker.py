import datetime

import numpy as np
from cloudmap_rs import SkyPixelQuery, AltAzCoord

from astroplan import Observer


class SunMasker:
    SUN_APPARENT_SIZE = np.rad2deg(0.54)

    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, sky_query: SkyPixelQuery, obs_time: datetime.datetime) -> SkyPixelQuery:
        sun_alt_az = self._observer.sun_altaz(obs_time)
        sky_query.mask_radius(AltAzCoord(sun_alt_az.alt.rad, sun_alt_az.az.rad), self.SUN_APPARENT_SIZE)

        return sky_query
