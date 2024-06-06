import datetime

import numpy as np
from astroplan import Observer
from numpy.typing import NDArray

import astropy.units as u


class SunAltDateFilter:
    def __init__(self, observer: Observer, threshold: float = -18):
        self._observer = observer
        self._threshold = threshold

    def __call__(self, dates: NDArray[datetime.datetime]) -> NDArray[datetime.datetime]:
        observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
        alt = observer.sun_altaz(np.array(dates)).alt.deg

        return dates[alt < self._threshold]
