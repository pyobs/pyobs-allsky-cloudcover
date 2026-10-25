import datetime

import numpy as np
import pytest

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.evaluation import best_threshold, evaluate
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.simulation.truth import TRUTH_ALT, TRUTH_AZ, GroundTruth
from pyobs_cloudcover.sky_query import SkyPixelQuery

T = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def truth_east_cloudy() -> GroundTruth:
    ext = np.zeros((len(TRUTH_ALT), len(TRUTH_AZ)))
    ext[:, TRUTH_AZ < 180] = 3.0
    return GroundTruth(ext, 0.5, {})


def result(values: np.ndarray, threshold: float) -> CloudCoverageInfo:
    grid = SphericalAltAzGenerator(20000, 30)()
    cloud = np.where(np.isnan(values), np.nan, (values < threshold).astype(float))
    return CloudCoverageInfo(SkyPixelQuery(grid, cloud), None, T, SkyPixelQuery(grid, values), "x", "night")


def east_values(clear: float = 6.0, cloudy: float = 3.0) -> np.ndarray:
    grid = SphericalAltAzGenerator(20000, 30)()
    return np.where(np.rad2deg(grid.az) < 180, cloudy, clear)


def test_perfect_result() -> None:
    m = evaluate(result(east_values(), 4.5), truth_east_cloudy())
    assert m is not None
    assert m.agreement == pytest.approx(1.0, abs=0.01)
    assert m.total_true == pytest.approx(0.5, abs=0.02)
    assert m.total_error < 0.02


def test_rethresholding() -> None:
    info = result(east_values(), 2.0)  # everything clear
    assert evaluate(info, truth_east_cloudy()).agreement == pytest.approx(0.5, abs=0.02)  # type: ignore[union-attr]
    t, score = best_threshold([(info, truth_east_cloudy())], np.arange(2, 7, 0.5))
    assert 3.0 < t <= 6.0 and score > 0.98


def test_masked_positions_dont_count() -> None:
    values = east_values()
    values[: len(values) // 2] = np.nan
    m = evaluate(result(values, 4.5), truth_east_cloudy())
    assert m is not None and m.agreement > 0.98 and 0.4 < m.coverage < 0.6
