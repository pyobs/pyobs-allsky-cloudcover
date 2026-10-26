from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


@dataclass
class StarMeasurement:
    signal: npt.NDArray[np.float64]  # background-subtracted aperture sum
    noise: npt.NDArray[np.float64]  # background noise of that sum
    background: npt.NDArray[np.float64]  # local background per pixel


class ApertureMatcher:
    """Decides for each catalogue star whether it is visible, with local aperture photometry.

    Around the predicted position, the background level and noise come from an annulus (median and MAD, so
    neighbouring stars hardly bias them). The brightest pixel within ``search_radius`` (allowing for
    world-model errors) is taken as the star's centre, and the background-subtracted sum over the pixels within
    ``aperture_radius`` of it must exceed ``snr`` times the background noise of that aperture. Stars whose
    background is above ``max_background`` (a fraction of the saturation level, e.g. moon glare) are not
    detectable and count as not seen.

    All radii are in pixels of the analysed (binned) image. Vectorised over all stars.
    """

    def __init__(
        self,
        snr: float = 5.0,
        search_radius: float = 2.0,
        aperture_radius: float = 1.5,
        annulus: tuple[float, float] = (4.0, 7.0),
        max_background: float = 0.8,
    ) -> None:
        self._snr = snr
        self._search_radius = search_radius
        self._aperture_radius = aperture_radius
        self._annulus = annulus
        self._max_background = max_background
        self.last: StarMeasurement | None = None

    def __call__(
        self, image: npt.NDArray[np.float64], catalog: PixelCatalog, saturation: float | None = None
    ) -> list[bool]:
        m = self.measure(image, catalog)
        self.last = m
        found = m.signal / m.noise > self._snr
        if saturation is not None:
            found &= ~(m.background > self._max_background * saturation)
        found &= np.isfinite(m.background) & np.isfinite(m.noise)
        return [bool(f) for f in found]

    def measure(self, image: npt.NDArray[np.float64], catalog: PixelCatalog) -> StarMeasurement:
        """Background-subtracted aperture sum, its background noise and the local background of every star."""
        n = len(catalog.px)
        if n == 0:
            empty = np.zeros(0)
            return StarMeasurement(empty, empty, empty)
        half = int(np.ceil(self._annulus[1]))
        padded = np.pad(np.asarray(image, dtype=np.float64), half, mode="constant", constant_values=np.nan)
        off = np.arange(-half, half + 1)
        ix = np.rint(catalog.px).astype(int)
        iy = np.rint(catalog.py).astype(int)
        rows = (iy + half)[:, None, None] + off[None, :, None]
        cols = (ix + half)[:, None, None] + off[None, None, :]
        rows = np.clip(rows, 0, padded.shape[0] - 1)
        cols = np.clip(cols, 0, padded.shape[1] - 1)
        cut = padded[rows, cols]  # (n, k, k)

        # distances from the predicted (sub-pixel) position
        dx = off[None, None, :] + ix[:, None, None] - np.asarray(catalog.px)[:, None, None]
        dy = off[None, :, None] + iy[:, None, None] - np.asarray(catalog.py)[:, None, None]
        r = np.hypot(dx, dy)

        ring = (r >= self._annulus[0]) & (r <= self._annulus[1])
        ring_values = np.where(ring, cut, np.nan).reshape(n, -1)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                background = np.nanmedian(ring_values, axis=1)
                sigma = 1.4826 * np.nanmedian(np.abs(ring_values - background[:, None]), axis=1)

        # brightest pixel near the predicted position
        search = np.where((r <= self._search_radius) & np.isfinite(cut), cut, -np.inf).reshape(n, -1)
        peak = np.argmax(search, axis=1)
        k = cut.shape[1]
        py, px = np.divmod(peak, k)
        # aperture around the peak
        ay = off[None, :, None] - off[py][:, None, None]
        ax = off[None, None, :] - off[px][:, None, None]
        aperture = np.hypot(ax, ay) <= self._aperture_radius
        n_ap = aperture.reshape(n, -1).sum(axis=1)
        signal = np.nansum(np.where(aperture, cut - background[:, None, None], 0.0).reshape(n, -1), axis=1)
        noise = np.maximum(sigma, 1e-9) * np.sqrt(n_ap)
        return StarMeasurement(signal, noise, background)
