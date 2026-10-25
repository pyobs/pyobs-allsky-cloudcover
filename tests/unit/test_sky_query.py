"""Ported from the unit tests of the former Rust extension (src/sky_query.rs)."""
import numpy as np
import pytest

from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery, angle_to_chord, chord_to_angle


def test_chord_roundtrip() -> None:
    for angle in (0.0, 0.1, 1.0, np.pi / 2, np.pi):
        assert chord_to_angle(angle_to_chord(angle)) == pytest.approx(angle)


def test_query_nearest_coordinate() -> None:
    query = SkyPixelQuery(SkyGrid([0.0, np.pi / 4], [0.0, np.pi]), [1.0, 0.0])
    assert query.query_nearest_coordinate(np.pi / 2, 0.0) == 0.0


def test_query_radius() -> None:
    query = SkyPixelQuery(SkyGrid([np.pi / 4, np.pi / 4], [0.0, np.pi]), [1.0, 0.0])
    assert query.query_radius(np.pi / 2, 0.0, np.pi / 2) == 0.5


def test_query_radius_none() -> None:
    query = SkyPixelQuery(SkyGrid([np.pi / 4, np.pi / 4], [0.0, np.pi]), [np.nan, np.nan])
    assert query.query_radius(np.pi / 2, 0.0, np.pi / 2) is None


def test_mask_radius() -> None:
    query = SkyPixelQuery(SkyGrid([np.pi / 4, np.pi / 4, -np.pi / 4], [0.0, np.pi, 0.0]), [1.0, 0.0, 1.0])
    query.mask_radius(np.pi / 2, 0.0, np.pi / 2)
    np.testing.assert_array_equal(np.isnan(query.get_pixels()), [True, True, False])


def test_azimuth_wraps() -> None:
    grid = SkyGrid([0.0], [np.deg2rad(359.5)])
    assert len(grid.within(0.0, np.deg2rad(0.5), np.deg2rad(1.01))) == 1


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        SkyPixelQuery(SkyGrid([0.0], [0.0]), [1.0, 0.0])
