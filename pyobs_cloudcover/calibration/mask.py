"""Build the sky mask (horizon, trees, buildings, the camera housing) from images."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from pyobs_cloudcover.calibration.astrometry import luminance
from pyobs_cloudcover.camera.bayer import debayer
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.world_model.world_model import WorldModel

BoolArray = npt.NDArray[np.bool_]


def sky_mask_from_day(
    frames: Sequence[Frame],
    model: WorldModel,
    min_alt: float = 0.0,
    darkness: float = 0.45,
    ring_width: float = 2.0,
    smooth: int = 3,
) -> BoolArray:
    """Sky mask (True = sky) from daytime frames.

    Obstructions are much darker than the sky at the same altitude, so each pixel is compared with the bright end
    (70th percentile of the unsaturated pixels, since trees may fill much of a low ring and the sun's glare
    would dominate its ring) of its altitude ring (``ring_width`` degrees), in its brightest colour (blue sky is
    dark in red); pixels below ``darkness`` times that are obstructed. Several frames
    (e.g. different times of day, clear and cloudy) are combined: a pixel is sky if it looks like sky in most of
    them. The result is cleaned morphologically; everything below ``min_alt`` or outside the lens is masked.
    """
    if not frames:
        raise ValueError("No frames")
    h, w = frames[0].shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    alt, _ = model.pix_to_altaz(x, y)
    alt = np.rad2deg(np.asarray(alt))
    usable = np.isfinite(alt) & (alt >= min_alt)
    rings = np.where(usable, (np.clip(alt, 0, 90) / ring_width).astype(int), -1)

    votes = np.zeros((h, w))
    for frame in frames:
        lum = ndimage.median_filter(_brightest_channel(frame), size=smooth)
        ring_median = np.zeros(rings.max() + 2)
        # the sun's saturated glare would make everything else in its ring look dark: leave it out
        unsaturated = lum < 0.9 * frame.saturation
        for r in np.unique(rings[usable]):
            ring = (rings == r) & unsaturated
            ring_median[r] = np.percentile(lum[ring], 70) if ring.sum() > 10 else np.inf
        reference = np.where(usable, ring_median[np.clip(rings, 0, None)], np.inf)
        votes += lum >= darkness * reference
    sky = (votes >= len(frames) / 2) & usable

    # tidy up: remove specks, fill small holes (e.g. a dark cloud gap), keep the big connected sky region
    sky = ndimage.binary_opening(sky, iterations=2)
    sky = ndimage.binary_closing(sky, iterations=2)
    labels, n = ndimage.label(sky)
    if n > 1:
        sizes = ndimage.sum(sky, labels, range(1, n + 1))
        sky = labels == (int(np.argmax(sizes)) + 1)
    return np.asarray(sky & usable)


def _brightest_channel(frame: Frame) -> npt.NDArray[np.float64]:
    """Per pixel the brightest colour: blue sky is dark in red but bright in blue, so in luminance it can look as
    dark as a tree next to sunlit cumulus; obstructions are dark in every colour."""
    if frame.bayer is not None and frame.data.ndim == 2:
        return np.asarray(debayer(frame.data, frame.bayer).max(axis=2), dtype=np.float64)
    if frame.data.ndim == 3:
        return np.asarray(frame.data.max(axis=2), dtype=np.float64)
    return luminance(frame)


def sky_mask_from_altitude(shape: tuple[int, int], model: WorldModel, min_alt: float) -> BoolArray:
    """Simple mask: everything above ``min_alt`` degrees."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    alt, _ = model.pix_to_altaz(x, y)
    alt = np.asarray(alt)
    return np.asarray(np.isfinite(alt) & (np.rad2deg(alt) >= min_alt))
