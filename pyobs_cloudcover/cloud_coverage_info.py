import datetime
from typing import Optional

from cloudmap_rs import SkyPixelQuery


class CloudCoverageInfo(object):
    def __init__(self, cloud_cover_query: SkyPixelQuery,
                 total_cover: float, zenith_cover: Optional[float], change: Optional[float], obs_time: datetime.datetime) -> None:
        self.cloud_cover_query = cloud_cover_query
        self.total_cover = total_cover
        self.zenith_cover = zenith_cover
        self.change = change
        self.obs_time = obs_time
