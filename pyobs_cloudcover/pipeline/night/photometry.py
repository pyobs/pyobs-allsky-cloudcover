"""Photometric mode: how many magnitudes of cloud extinction lie in front of each star, and a map of it.

The instrumental magnitude of a star, ``m = -2.5 log10(flux / exposure)``, relates to its catalogue magnitude by
``m = V + k X + A - Z``: ``Z`` the camera's zero point, ``k`` the atmospheric extinction per airmass ``X``, ``A`` the
extinction by cloud. Per frame, ``Z`` comes from the stars with the least extinction (a high percentile of
``V + k X - m``). It is remembered across frames, so an overcast frame doesn't drag it down, and slowly allowed to
fall in case the camera really got less sensitive (dew, dirt). Stars that aren't detected give a lower limit on
their ``A``.

Unlike the limiting magnitude, this doesn't depend on how bright the sky is: moonlight or twilight make faint stars
harder to detect but don't dim the stars that are detected. Thin cloud shows up as a fraction of a magnitude
instead of all or nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.sky_query import SkyGrid, angle_to_chord, chord_to_angle
from pyobs_cloudcover.world_model.world_model import WorldModel

FloatArray = npt.NDArray[np.float64]
ARCSEC2_PER_SR = (180 / np.pi * 3600) ** 2


def airmass(alt: FloatArray) -> FloatArray:
    z = np.pi / 2 - np.clip(alt, np.deg2rad(1), None)
    return np.asarray((1 - 0.96 * np.sin(z) ** 2) ** -0.5)


@dataclass
class StarExtinction:
    extinction: FloatArray  # mag, NaN where the star wasn't detected
    lower_limit: FloatArray  # mag, for undetected stars (NaN for detected ones)
    zero_point: float  # this frame's estimate
    reference_zero_point: float  # the one used


class PhotometricCalibrator:
    """Zero point and per-star cloud extinction. Keeps the zero point between frames (it's a camera property).

    Args:
        extinction_coefficient: Atmospheric extinction (mag per airmass) of the site, ~0.15-0.3 in V.
        zero_point: Fixed zero point (mag giving 1 ADU/s in the analysed image); default: measured.
        snr: Detection threshold for a star's flux to count as a measurement.
        decay: How fast (mag per frame) the remembered zero point may drop when frames suggest a lower one.
        bright: Stars up to this magnitude determine the zero point.
        usable_snr: Stars are used for extinction if they would have at least this SNR without cloud.
    """

    def __init__(
        self,
        extinction_coefficient: float = 0.2,
        zero_point: float | None = None,
        snr: float = 5.0,
        decay: float = 0.01,
        bright: float = 4.5,
        usable_snr: float = 8.0,
        max_jump: float = 0.3,
    ) -> None:
        self._k = extinction_coefficient
        self._fixed = zero_point
        self._snr = snr
        self._decay = decay
        self._bright = bright
        self._usable_snr = usable_snr
        self._max_jump = max_jump
        self.reference: float | None = zero_point

    def __call__(
        self,
        signal: FloatArray,
        noise: FloatArray,
        v_mag: FloatArray,
        alt: FloatArray,
        exposure: float,
        may_learn: bool = True,
    ) -> StarExtinction:
        """``may_learn``: this frame may set or raise the remembered zero point (pass False if it looks overcast
        by other means, so a uniformly dimmed frame isn't taken for clear)."""
        expected = v_mag + self._k * airmass(alt)  # V + kX
        detected = (signal / noise > self._snr) & (signal > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = -2.5 * np.log10(signal / exposure)
            m_upper = -2.5 * np.log10((np.maximum(signal, 0) + 3 * noise) / exposure)
        z_star = np.where(detected, expected - m, np.nan)

        # Zero point from bright, well-measured stars only: faint ones pick up neighbours and noise peaks and read
        # too bright. Clear stars agree closely; clouds dim stars and bright cloud structure (e.g. moonlit cirrus)
        # can make others look brighter, both scattered. So: the densest cluster of the per-star values, if most of
        # the well-measured stars agree. (A first frame under uniform overcast would still pass: set zero_point.)
        good = detected & (signal / noise > 4 * self._snr) & (v_mag <= self._bright)
        frame_zp = float("nan")
        if np.sum(good) >= 8:
            centre, members = _densest(z_star[good], width=0.3)
            if members >= 0.5 * np.sum(good):
                frame_zp = centre
        if self._fixed is None and np.isfinite(frame_zp) and (may_learn or self.reference is not None):
            if self.reference is None:
                self.reference = frame_zp
            elif not may_learn:
                self.reference = max(min(frame_zp, self.reference), self.reference - self._decay)
            elif frame_zp > self.reference + self._max_jump:
                # a camera doesn't suddenly get much more sensitive: follow only slowly
                self.reference += self._decay
            elif frame_zp > self.reference:
                self.reference = frame_zp
            else:
                self.reference = max(frame_zp, self.reference - self._decay)
        zp = self.reference if self.reference is not None else (frame_zp if may_learn else float("nan"))

        # Only stars that should be clearly visible without cloud say something unbiased about extinction; the
        # noise (sky brightness, moonlight) decides which those are.
        with np.errstate(over="ignore", invalid="ignore"):
            expected_snr = 10 ** (-0.4 * (expected - zp)) * exposure / noise
        usable = expected_snr >= self._usable_snr
        extinction = np.where(detected & usable, zp - z_star, np.nan)
        # undetected: fainter than the 3-sigma upper limit, so at least this much extinction
        lower = np.where(~detected & usable, zp - expected + m_upper, np.nan)
        return StarExtinction(extinction, lower, frame_zp, float(zp) if zp is not None else float("nan"))


def _densest(values: FloatArray, width: float) -> tuple[float, int]:
    """Centre (mean) of the densest ``width``-wide cluster of values, and how many values it holds."""
    v = np.sort(values[np.isfinite(values)])
    if len(v) == 0:
        return float("nan"), 0
    hi = np.searchsorted(v, v + width, side="right")
    best = int(np.argmax(hi - np.arange(len(v))))
    return float(np.mean(v[best : hi[best]])), int(hi[best] - best)


def extinction_map(
    grid: SkyGrid,
    star_alt: FloatArray,
    star_az: FloatArray,
    stars: StarExtinction,
    radius: float,
    max_extinction: float = 3.0,
) -> FloatArray:
    """Cloud extinction (mag) at every grid position: weighted mean over the stars within ``radius`` (rad), weights
    ``exp(-d/radius)``. Detected stars give their extinction, undetected ones their lower limit when that says
    something (> 0); both clipped to [0, ``max_extinction``]. NaN without any informative star."""
    value = np.where(np.isfinite(stars.extinction), stars.extinction, stars.lower_limit)
    informative = np.isfinite(value) & (np.isfinite(stars.extinction) | (value > 0))
    out = np.full(len(grid), np.nan)
    if not np.any(informative) or len(grid) == 0:
        return out
    sky = SkyGrid(star_alt[informative], star_az[informative])
    pairs = grid.tree.sparse_distance_matrix(sky.tree, angle_to_chord(radius), output_type="ndarray")
    if len(pairs) == 0:
        return out
    w = np.exp(-np.asarray(chord_to_angle(pairs["v"])) / radius)
    v = np.clip(value[informative][pairs["j"]], 0, max_extinction)
    wsum = np.bincount(pairs["i"], w, minlength=len(grid))
    vsum = np.bincount(pairs["i"], w * v, minlength=len(grid))
    has = wsum > 0
    out[has] = vsum[has] / wsum[has]
    return out


def sky_brightness(
    image: FloatArray,
    model: WorldModel,
    zero_point: float,
    exposure: float,
    min_alt: float = 75.0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float | None:
    """Sky surface brightness (V mag/arcsec^2) around the zenith: median background above ``min_alt`` per pixel
    solid angle, with the photometric zero point."""
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    alt, _ = model.pix_to_altaz(x, y)
    alt = np.asarray(alt)
    sel = np.isfinite(alt) & (alt >= np.deg2rad(min_alt))
    if mask is not None:
        sel &= mask
    if sel.sum() < 20 or not np.isfinite(zero_point):
        return None
    level = float(np.median(image[sel]))
    # bias and dark: the level outside the lens circle, if the image shows any of it
    outside = ~np.isfinite(alt)
    if outside.sum() > 100:
        level -= float(np.median(image[outside]))
    # solid angle of a pixel near the zenith, from the model
    cy, cx = np.median(y[sel]), np.median(x[sel])
    a0, z0 = model.pix_to_altaz(np.array([cx]), np.array([cy]))
    a1, z1 = model.pix_to_altaz(np.array([cx + 1]), np.array([cy]))
    a2, z2 = model.pix_to_altaz(np.array([cx]), np.array([cy + 1]))

    def vec(a: Any, z: Any) -> FloatArray:
        a, z = float(np.asarray(a)[0]), float(np.asarray(z)[0])
        return np.array([np.cos(a) * np.cos(z), np.cos(a) * np.sin(z), np.sin(a)])

    v0, v1, v2 = vec(a0, z0), vec(a1, z1), vec(a2, z2)
    omega = float(np.linalg.norm(np.cross(v1 - v0, v2 - v0))) * ARCSEC2_PER_SR
    if level <= 0 or omega <= 0:
        return None
    return float(-2.5 * np.log10(level / exposure / omega) + zero_point)
