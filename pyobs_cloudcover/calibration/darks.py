"""Master darks and hot-pixel maps (raw orientation, as the camera profile applies them)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy import ndimage

from pyobs_cloudcover.camera.reader import RawImage

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def master_dark(darks: Sequence[RawImage]) -> tuple[FloatArray, float | None]:
    """Median of dark frames (lens covered, same exposure and temperature as the night frames), and its exposure."""
    if not darks:
        raise ValueError("No dark frames")
    stack = np.median(np.stack([d.data.astype(np.float64) for d in darks]), axis=0)
    exposures = {d.header.get("EXPTIME", d.header.get("EXPOSURE")) for d in darks}
    single = exposures.pop() if len(exposures) == 1 else None
    exposure = float(single) if single is not None else None
    return np.asarray(stack), exposure


def write_dark(path: str | Path, dark: FloatArray, exposure: float | None) -> None:
    header = fits.Header()
    header["IMAGETYP"] = "Master Dark"
    if exposure is not None:
        header["EXPTIME"] = exposure
    fits.PrimaryHDU(dark.astype(np.float32), header=header).writeto(path, overwrite=True)


def _quantum(data: FloatArray) -> float:
    """Smallest step between stored values (1 ADU, or 16 for 12-bit data shifted into 16 bits)."""
    sample = np.unique(data.ravel()[:: max(1, data.size // 200000)])
    steps = np.diff(sample)
    return float(np.min(steps[steps > 0])) if np.any(steps > 0) else 1.0


def _sharp_excess(data: FloatArray) -> FloatArray:
    """How far each pixel stands out from its same-colour neighbours (2 px away, so it works for Bayer mosaics),
    in units of the robust noise, counting only pixels that stand out *alone*: a hot pixel's direct neighbours are
    at background level, a star's are lit too."""
    excess = np.zeros_like(data)
    for dy in (0, 1):
        for dx in (0, 1):
            plane = data[dy::2, dx::2]
            excess[dy::2, dx::2] = plane - ndimage.median_filter(plane, size=3, mode="nearest")
    sigma = max(1.4826 * float(np.median(np.abs(excess - np.median(excess)))), _quantum(data))
    # light in the 4 direct neighbours above their own local level
    neighbours = ndimage.convolve(excess, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0, mode="nearest")
    sharp = neighbours < 0.25 * np.maximum(excess, 0)
    return np.asarray(np.where(sharp, excess, 0.0) / sigma)


def hot_pixels_from_dark(dark: FloatArray, sigma: float = 8.0) -> BoolArray:
    """Pixels of a master dark that stand out from their neighbours by more than ``sigma``."""
    return np.asarray(_sharp_excess(np.asarray(dark, dtype=np.float64)) > sigma)


def hot_pixels_from_frames(frames: Sequence[RawImage], sigma: float = 6.0, fraction: float = 0.8) -> BoolArray:
    """Hot pixels from ordinary night frames: pixels that stand out from their neighbours in at least ``fraction``
    of the frames. Stars move between frames (use frames at least ~15 minutes apart), hot pixels don't."""
    if len(frames) < 3:
        raise ValueError("Need at least 3 frames, taken well apart in time")
    count = np.zeros(frames[0].data.shape[:2])
    for f in frames:
        data = f.data.astype(np.float64)
        if data.ndim == 3:
            data = data.mean(axis=2)
        count += _sharp_excess(data) > sigma
    return np.asarray(count >= fraction * len(frames))
