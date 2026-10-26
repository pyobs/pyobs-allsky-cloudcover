import numpy as np

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.field_evaluators.field_evaluator import FieldEvaluator


class SkyBrightnessField(FieldEvaluator):
    """Zenith sky brightness (V mag/arcsec^2), like pyobs' SKYMAG sensor (photometric mode, night only)."""

    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        return cloud_info.sky_brightness


class ZeroPointField(FieldEvaluator):
    """Photometric zero point of the image (mag giving 1 ADU/s), e.g. to watch for dew or a dirty dome."""

    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        return cloud_info.zero_point


class MapMeanField(FieldEvaluator):
    """Mean of a continuous map (limiting magnitude or cloud extinction) above ``altitude`` degrees."""

    def __init__(self, which: str, altitude: float) -> None:
        self._which = which
        self._radius = np.deg2rad(90.0 - altitude)

    def __call__(self, cloud_info: CloudCoverageInfo) -> float | None:
        if self._which == "extinction":
            query = cloud_info.extinction_query
        elif cloud_info.value_name == "limiting_magnitude":
            query = cloud_info.value_query
        else:
            query = None
        return None if query is None else query.query_radius(np.pi / 2, 0.0, self._radius)
