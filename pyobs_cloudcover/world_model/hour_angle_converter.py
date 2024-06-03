import datetime

from astroplan import Observer
from astropy.coordinates import Angle


class HourAngleConverter(object):
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, ra: Angle, obs_time: datetime.datetime) -> float:
        lst = self._observer.local_sidereal_time(obs_time)
        self._lst = Angle(lst, unit="hourangle")

        return float((self._lst - ra).rad)