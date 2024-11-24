import datetime
from typing import Optional

from cloudmap_rs import SkyPixelQuery


class CloudCoverageInfo(object):
    def __init__(self, cloud_cover_query: SkyPixelQuery, change: Optional[float], cloud_plot: bytes, obs_time: datetime.datetime) -> None:
        self.cloud_cover_query = cloud_cover_query
        self.change = change
        self.cloud_plot = cloud_plot
        self.obs_time = obs_time
