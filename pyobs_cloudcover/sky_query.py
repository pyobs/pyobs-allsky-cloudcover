from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

FloatArray = npt.NDArray[np.float64]
ArrayLike = FloatArray | float


def altaz_to_xyz(alt: npt.ArrayLike, az: npt.ArrayLike) -> FloatArray:
    """Unit vectors (..., 3) for altitude/azimuth in rad."""
    alt, az = np.asarray(alt, dtype=np.float64), np.asarray(az, dtype=np.float64)
    cos_alt = np.cos(alt)
    return np.stack([cos_alt * np.cos(az), cos_alt * np.sin(az), np.sin(alt)], axis=-1)


def angle_to_chord(angle: float) -> float:
    """Straight-line distance between two unit vectors separated by ``angle`` (rad)."""
    return float(2 * np.sin(min(max(angle, 0.0), np.pi) / 2))


def chord_to_angle(chord: ArrayLike) -> ArrayLike:
    return 2 * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


class SkyGrid:
    """A fixed set of positions on the sky with a spatial index, built once and reused for many maps."""

    def __init__(self, alt: npt.ArrayLike, az: npt.ArrayLike) -> None:
        self.alt: FloatArray = np.asarray(alt, dtype=np.float64).ravel()
        self.az: FloatArray = np.asarray(az, dtype=np.float64).ravel()
        self.xyz = altaz_to_xyz(self.alt, self.az)

    def __len__(self) -> int:
        return len(self.alt)

    @functools.cached_property
    def tree(self) -> cKDTree:
        return cKDTree(self.xyz)

    def nearest(self, alt: float, az: float) -> int | None:
        if len(self) == 0:
            return None
        _, index = self.tree.query(altaz_to_xyz(alt, az))
        return int(index)

    def within(self, alt: float, az: float, radius: float) -> npt.NDArray[np.intp]:
        """Indices of all grid positions within ``radius`` (rad, great-circle) of the given position."""
        if len(self) == 0:
            return np.array([], dtype=np.intp)
        return np.asarray(self.tree.query_ball_point(altaz_to_xyz(alt, az), angle_to_chord(radius)), dtype=np.intp)


class SkyPixelQuery:
    """Values on a :class:`SkyGrid`. ``NaN`` marks unknown/masked positions.

    For cloud maps the values are 1.0 (cloudy), 0.0 (clear) or NaN.
    """

    def __init__(self, grid: SkyGrid, pixels: npt.ArrayLike) -> None:
        self.grid = grid
        self._pixels: FloatArray = np.array(pixels, dtype=np.float64).ravel()

        if len(self._pixels) != len(grid):
            raise ValueError(f"Got {len(self._pixels)} values for {len(grid)} grid positions.")

    def set_pixels(self, pixels: npt.ArrayLike) -> None:
        self._pixels = np.array(pixels, dtype=np.float64).ravel()

    def get_pixels(self) -> FloatArray:
        return self._pixels.copy()

    def query_nearest_coordinate(self, alt: float, az: float) -> float | None:
        index = self.grid.nearest(alt, az)
        if index is None or np.isnan(self._pixels[index]):
            return None
        return float(self._pixels[index])

    def query_radius(self, alt: float, az: float, radius: float) -> float | None:
        """Mean of the known values within ``radius`` (rad); None if there are none."""
        values = self._pixels[self.grid.within(alt, az, radius)]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return None
        return float(np.mean(values))

    def mask_radius(self, alt: float, az: float, radius: float) -> None:
        self._pixels[self.grid.within(alt, az, radius)] = np.nan
