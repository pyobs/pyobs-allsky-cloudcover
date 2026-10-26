import numpy as np

from pyobs_cloudcover.pipeline.night.photometry import (
    PhotometricCalibrator,
    StarExtinction,
    airmass,
    extinction_map,
)
from pyobs_cloudcover.sky_query import SkyGrid

ZP, K, EXPOSURE = 8.0, 0.2, 10.0


def stars(n: int = 200, cloud: float = 0.0, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    v = rng.uniform(1, 6, n)
    alt = np.deg2rad(rng.uniform(30, 90, n))
    flux = 10 ** (-0.4 * (v + K * airmass(alt) + cloud - ZP)) * EXPOSURE
    noise = np.full(n, 20.0)
    return flux + rng.normal(0, 20, n), noise, v, alt


def test_clear_sky_gives_zero_point_and_no_extinction() -> None:
    signal, noise, v, alt = stars()
    result = PhotometricCalibrator(K)(signal, noise, v, alt, EXPOSURE)
    assert abs(result.reference_zero_point - ZP) < 0.1
    detected = np.isfinite(result.extinction)
    assert detected.sum() > 100
    assert abs(float(np.nanmedian(result.extinction))) < 0.1


def test_zero_point_is_remembered_through_cloud() -> None:
    calibrator = PhotometricCalibrator(K)
    calibrator(*stars(seed=1), EXPOSURE)
    result = calibrator(*stars(cloud=1.5, seed=2), EXPOSURE)
    assert abs(result.reference_zero_point - ZP) < 0.15
    assert abs(float(np.nanmedian(result.extinction)) - 1.5) < 0.2
    # stars lost in the cloud say at least something
    assert np.nanmax(result.lower_limit) > 1.0


def test_extinction_map() -> None:
    grid = SkyGrid(np.deg2rad([60.0, 60.0]), np.deg2rad([90.0, 270.0]))
    alt = np.deg2rad([58.0, 62.0, 58.0, 62.0])
    az = np.deg2rad([88.0, 92.0, 268.0, 272.0])
    ext = StarExtinction(np.array([0.0, 0.1, 2.0, np.nan]), np.array([np.nan, np.nan, np.nan, 1.8]), ZP, ZP)
    result = extinction_map(grid, alt, az, ext, np.deg2rad(5))
    assert result[0] < 0.2 and result[1] > 1.5
