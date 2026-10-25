import datetime

from pyobs_cloudcover.sky_query import SkyPixelQuery


class CloudCoverageInfo:
    def __init__(self, cloud_cover_query: SkyPixelQuery, change: float | None, obs_time: datetime.datetime) -> None:
        self.cloud_cover_query = cloud_cover_query
        self.change = change
        self.obs_time = obs_time
