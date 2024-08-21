import datetime

import numpy as np
from cloudmap_rs import SkyPixelQuery, AltAzCoord

from astroplan import Observer


class MoonMasker:
    def __init__(self, moon_apparent_size: float, observer: Observer):
        self._moon_apparent_size = moon_apparent_size
        self._observer = observer

    def __call__(self, sky_query: SkyPixelQuery, obs_time: datetime.datetime) -> SkyPixelQuery:
        moon_alt_az = self._observer.moon_altaz(obs_time)
        sky_query.mask_radius(AltAzCoord(moon_alt_az.alt.rad, moon_alt_az.az.rad), np.deg2rad(self._moon_apparent_size))

        return sky_query
