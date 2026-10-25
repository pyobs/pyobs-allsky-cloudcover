import datetime
from typing import cast

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from astropy.coordinates import Angle


class HourAngleConverter:
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, ra: Angle, obs_time: datetime.datetime) -> float | npt.NDArray[np.float64]:
        lst = self._observer.local_sidereal_time(obs_time)
        self._lst = Angle(lst, unit="hourangle")

        return cast(float | npt.NDArray[np.float64], (self._lst - ra).rad)
