import datetime
from typing import Union, cast

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from astropy.coordinates import Angle


class HourAngleConverter(object):
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, ra: Angle, obs_time: datetime.datetime) -> Union[float, npt.NDArray[np.float_]]:
        lst = self._observer.local_sidereal_time(obs_time)
        self._lst = Angle(lst, unit="hourangle")

        return cast(Union[float, npt.NDArray[np.float_]], (self._lst - ra).rad)