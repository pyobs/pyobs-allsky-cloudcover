from typing import Optional, cast

import numpy as np
from astroplan import Observer
from cloudmap_rs import AltAzCoord

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class SunCloudCoverageCalculator(FieldEvaluator):
    def __init__(self, observer: Observer, radius: float):
        self._observer = observer
        self._radius = radius

    def __call__(self, cloud_info: CloudCoverageInfo) -> Optional[float]:
        sun_alt_az = self._observer.sun_altaz(cloud_info.obs_time)
        coverage = cloud_info.cloud_cover_query.query_radius(
            AltAzCoord(sun_alt_az.alt.rad, sun_alt_az.az.rad),
            np.deg2rad(self._radius)
        )

        return cast(Optional[float], coverage)
