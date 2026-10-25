from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model.world_model import WorldModel

FloatArray = npt.NDArray[np.float64]

# r / f as a function of the zenith angle theta, and its inverse
_PROJECTIONS: dict[str, tuple[Callable[[FloatArray], FloatArray], Callable[[FloatArray], FloatArray]]] = {
    "equidistant": (lambda t: t, lambda r: r),
    "equisolid": (lambda t: 2 * np.sin(t / 2), lambda r: 2 * np.arcsin(r / 2)),
    "stereographic": (lambda t: 2 * np.tan(t / 2), lambda r: 2 * np.arctan(r / 2)),
    "orthographic": (np.sin, np.arcsin),
}


class FisheyeModel(WorldModel):
    """Radially symmetric fisheye lens looking at the zenith.

    A sky position at zenith angle ``theta`` and azimuth ``az`` lands at

        x = c_x + r(theta) * cos(parity * (az - a0))
        y = c_y + r(theta) * sin(parity * (az - a0))

    in (canonical, full-resolution) pixel coordinates, where ``r(theta) = f * g(theta)`` for one of the
    standard projections ``g`` (equidistant, equisolid, stereographic, orthographic), or, for
    ``projection="polynomial"``, ``r(theta) = sum(coefficients[i] * theta**(i+1))``. With ``parity=1`` and an
    equisolid projection this is the same mapping as :class:`SimpleModel` with ``F=2``.

    Args:
        c_x, c_y: Pixel position of the zenith.
        f: Focal length in pixels (pixels per radian at the zenith); ignored for polynomial projections.
        a0: Azimuth (rad) that points along the +x axis (for parity=1).
        projection: Lens projection.
        coefficients: Polynomial coefficients in pixels, for ``projection="polynomial"``.
        parity: +1 or -1; -1 mirrors the image (azimuth runs the other way round).
        max_zenith_angle: Largest zenith angle (deg) the lens maps; pixels further out have no sky position.
    """

    def __init__(
        self,
        c_x: float,
        c_y: float,
        f: float = 1.0,
        a0: float = 0.0,
        projection: str = "equisolid",
        coefficients: Sequence[float] | None = None,
        parity: int = 1,
        max_zenith_angle: float = 100.0,
    ) -> None:
        if parity not in (1, -1):
            raise ValueError("parity must be 1 or -1")
        if projection != "polynomial" and projection not in _PROJECTIONS:
            raise ValueError(f"Unknown projection {projection}, use one of {[*_PROJECTIONS, 'polynomial']}")
        if projection == "polynomial" and not coefficients:
            raise ValueError("Polynomial projection needs coefficients")

        self.c_x, self.c_y, self.f, self.a0 = float(c_x), float(c_y), float(f), float(a0)
        self.projection = projection
        self.coefficients = [float(c) for c in coefficients] if coefficients else []
        self.parity = parity
        self.max_zenith_angle = float(max_zenith_angle)

        # orthographic lenses can't map beyond 90 deg
        self._theta_max = np.deg2rad(min(self.max_zenith_angle, 90.0 if projection == "orthographic" else 180.0))
        if projection == "polynomial":
            # tabulate r(theta) for the inverse; must be monotonic over the mapped range
            self._theta_table = np.linspace(0, self._theta_max, 4001)
            self._r_table = self._radius(self._theta_table)
            if np.any(np.diff(self._r_table) <= 0):
                raise ValueError("Polynomial projection must be strictly increasing up to max_zenith_angle")

    def _radius(self, theta: FloatArray) -> FloatArray:
        if self.projection == "polynomial":
            return sum(c * theta ** (i + 1) for i, c in enumerate(self.coefficients))  # type: ignore[return-value]
        return self.f * _PROJECTIONS[self.projection][0](theta)

    def _zenith_angle(self, r: FloatArray) -> FloatArray:
        r_max = self._radius(np.array(self._theta_max))
        with np.errstate(invalid="ignore"):
            if self.projection == "polynomial":
                theta = np.interp(r, self._r_table, self._theta_table, right=np.nan)
            else:
                theta = _PROJECTIONS[self.projection][1](r / self.f)
        return np.where(r <= r_max, theta, np.nan)

    def pix_to_altaz(
        self, x: FloatArray | float, y: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        dx, dy = np.asarray(x, dtype=np.float64) - self.c_x, np.asarray(y, dtype=np.float64) - self.c_y
        theta = self._zenith_angle(np.hypot(dx, dy))
        az = np.mod(self.a0 + self.parity * np.arctan2(dy, dx), 2 * np.pi)
        alt = np.pi / 2 - theta
        if np.ndim(alt) == 0:
            return float(alt), float(az)
        return alt, az

    def altaz_to_pix(
        self, alt: FloatArray | float, az: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        theta = np.pi / 2 - np.asarray(alt, dtype=np.float64)
        r = np.where(theta <= self._theta_max, self._radius(theta), np.nan)
        phi = self.parity * (np.asarray(az, dtype=np.float64) - self.a0)
        x, y = self.c_x + r * np.cos(phi), self.c_y + r * np.sin(phi)
        if np.ndim(x) == 0:
            return float(x), float(y)
        return x, y

    def scaled(self, factor: float) -> FisheyeModel:
        """The same lens on an image resampled by ``factor`` (e.g. 0.5 for 2x2 binning).

        Pixel centres are at integer coordinates, so a binned pixel i covers original pixels
        [i/factor - 0.5, (i+1)/factor - 0.5].
        """
        offset = 0.5 * (1 - factor)
        return FisheyeModel(
            c_x=self.c_x * factor - offset,
            c_y=self.c_y * factor - offset,
            f=self.f * factor,
            a0=self.a0,
            projection=self.projection,
            coefficients=[c * factor for c in self.coefficients] or None,
            parity=self.parity,
            max_zenith_angle=self.max_zenith_angle,
        )

    def to_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "class": "pyobs_cloudcover.world_model.FisheyeModel",
            "c_x": self.c_x,
            "c_y": self.c_y,
            "a0": self.a0,
            "projection": self.projection,
            "parity": self.parity,
            "max_zenith_angle": self.max_zenith_angle,
        }
        if self.projection == "polynomial":
            config["coefficients"] = list(self.coefficients)
        else:
            config["f"] = self.f
        return config
