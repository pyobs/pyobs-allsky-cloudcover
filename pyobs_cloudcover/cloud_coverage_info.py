import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt


class CloudCoverageInfo(object):
    def __init__(self, cloud_cover_map: npt.NDArray[np.float_],
                 total_cover: float, average: float, std: float, zenith_cover: float, zenith_average: float, zenith_std: float, change: Optional[float], obs_time: datetime.datetime) -> None:
        self.cloud_cover_map = cloud_cover_map
        self.total_cover = total_cover
        self.average = average
        self.std = std
        self.zenith_cover = zenith_cover
        self.zenith_average = zenith_average
        self.zenith_std = zenith_std
        self.change = change
        self.obs_time = obs_time
