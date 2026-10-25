import datetime

from pyobs_cloudcover.sky_query import SkyPixelQuery


class CloudCoverageInfo:
    """Result of analysing one image.

    Attributes:
        cloud_cover_query: Cloud map (1.0 cloudy, 0.0 clear, NaN unknown) on the pipeline's sky grid.
        change: Fraction of positions whose state changed since the previous image, if known.
        obs_time: Observation time (aware UTC).
        value_query: The continuous map the cloud map was thresholded from, if any: the limiting magnitude
            (night) or the colour ratio (day).
        value_name: What ``value_query`` holds: ``"limiting_magnitude"`` or ``"color_ratio"``.
        pipeline: Name of the pipeline that produced the result (``"night"``, ``"day"``, ...).
    """

    def __init__(
        self,
        cloud_cover_query: SkyPixelQuery,
        change: float | None,
        obs_time: datetime.datetime,
        value_query: SkyPixelQuery | None = None,
        value_name: str = "",
        pipeline: str = "",
    ) -> None:
        self.cloud_cover_query = cloud_cover_query
        self.change = change
        self.obs_time = obs_time
        self.value_query = value_query
        self.value_name = value_name
        self.pipeline = pipeline
