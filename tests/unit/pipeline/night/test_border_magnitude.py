"""Border-magnitude rules, ported from the unit tests of the former Rust extension (src/night/)."""

import numpy as np
import pytest

from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.border_magnitude import (
    border_magnitude,
    legacy_weight,
    robust_border_magnitude,
)
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
    result = border_magnitude(
        grid,
        np.array([np.pi / 4] * 2),
        np.array([0.0, np.pi]),
        np.array([1.0, 0.0]),
        np.array([False, True]),
        np.pi / 2,
        legacy_weight,
    )
    assert result[0] == pytest.approx(0.5)


def test_nearer_stars_weigh_more() -> None:
    # regression: the legacy weight grew with distance, so the farther star dominated
    grid = SkyGrid([np.pi / 2], [0.0])
    alt = np.deg2rad([88.0, 84.0])
    result = border_magnitude(
        grid, alt, np.array([0.0, np.pi]), np.array([5.0, 6.0]), np.array([True, False]), np.deg2rad(7)
    )
    assert 5.0 < result[0] < 5.5
    legacy = border_magnitude(
        grid, alt, np.array([0.0, np.pi]), np.array([5.0, 6.0]), np.array([True, False]), np.deg2rad(7), legacy_weight
    )
    assert legacy[0] > 5.5


def run_robust(mags: list[float], found: list[bool]) -> float:
    n = len(mags)
    alt = np.full(n, np.deg2rad(88.0))
    az = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return float(robust_border_magnitude(ZENITH, alt, az, np.array(mags), np.array(found), np.deg2rad(5), uniform)[0])


def test_robust_clean_border() -> None:
    assert run_robust([1.0, 2.0, 3.0, 4.0, 5.0], [True, True, True, False, False]) == 3.5


def test_robust_ignores_single_outliers() -> None:
    mags = [1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    found = [True, True, True, True, False, False, False, False, True, False]  # one false detection at 6.0
    assert run_robust(mags, found) == 3.75
    # the plain border rule is dragged towards the outlier
    assert run(mags, found) > 4.5


def test_robust_edge_cases() -> None:
    assert run_robust([2.0, 3.0], [True, True]) == 3.0
    assert run_robust([2.0, 3.0], [False, False]) == 2.0
    far = robust_border_magnitude(ZENITH, np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([True]), 0.1)
    assert np.isnan(far[0])


def test_robust_middle_of_equally_good_cuts() -> None:
    # seen at 2, unseen at 3, seen at 4, unseen at 5: cuts after 2 and after 4 both misclassify one star
    assert run_robust([2.0, 3.0, 4.0, 5.0], [True, False, True, False]) == (2.5 + 4.5) / 2
