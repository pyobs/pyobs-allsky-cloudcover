"""Border-magnitude rules, ported from the unit tests of the former Rust extension (src/night/)."""
import numpy as np
import pytest

from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.border_magnitude import border_magnitude
from pyobs_cloudcover.sky_query import SkyGrid

ZENITH = SkyGrid([np.pi / 2], [0.0])


def uniform(d: np.ndarray) -> np.ndarray:
    return np.ones_like(d)


def run(mags: list[float], found: list[bool]) -> float:
    n = len(mags)
    alt = np.full(n, np.deg2rad(88.0))
    az = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return float(border_magnitude(ZENITH, alt, az, np.array(mags), np.array(found), np.deg2rad(5), uniform)[0])


def test_no_overlap_uses_border_stars() -> None:
    # faintest seen 3.0, brightest unseen 4.0
    assert run([3.0, 3.0, 4.0, 5.0], [True, True, False, False]) == 3.5


def test_overlap_averages_all_stars_in_between() -> None:
    # faintest seen 4.0, brightest unseen 3.0 -> stars in [3, 4]: 4.0, 3.0, 3.0
    assert run([4.0, 3.0, 3.0, 5.0], [True, True, False, False]) == np.mean([4.0, 3.0, 3.0])


def test_only_seen_or_only_unseen() -> None:
    assert run([2.0, 3.0], [True, True]) == 3.0
    assert run([2.0, 3.0], [False, False]) == 2.0


def test_no_stars_is_nan() -> None:
    far = border_magnitude(ZENITH, np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([True]), 0.1, uniform)
    assert np.isnan(far[0])


def test_matches_rust_two_star_case() -> None:
    # src/night/cloud_map_generator.rs::test_gen_cloud_map: stars at 45 deg alt, query at zenith
    grid = SkyGrid([np.pi / 2], [0.0])
    result = border_magnitude(grid, np.array([np.pi / 4] * 2), np.array([0.0, np.pi]),
                              np.array([1.0, 0.0]), np.array([False, True]), np.pi / 2)
    assert result[0] == pytest.approx(0.5)
