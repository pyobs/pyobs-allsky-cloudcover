"""Keep the world model aligned with the sky while the camera moves a little (bumps, thermal flexure).

A 0.5 deg rotation of a wide-angle camera moves stars at the image edge by ~20 px, far beyond the detector's
search radius, so every star counts as unseen and a clear sky reads as cloudy. Before each night frame is
analysed, the aligner detects bright stars and fits a small sky rotation (:class:`AlignedModel`):

1. From the current correction, bright catalogue stars are matched within ``tolerance`` (deg); if enough match,
   the rotation is refined by least squares (tracking slow drifts).
2. Otherwise a grid of rotations within ``max_offset`` deg is scored by how many bright catalogue stars land on a
   detection, and the best is refined (recovering from a bump).
3. If neither gives ``min_stars`` matches with at least ``min_fraction`` of the bright stars in the image (e.g.
   cloudy), the correction is left as it was.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from scipy import optimize
from scipy.spatial import cKDTree

from pyobs_cloudcover.calibration.astrometry import catalog_altaz, detect_stars
from pyobs_cloudcover.world_model.aligned_model import AlignedModel, altaz_to_enu, enu_to_altaz, small_rotation

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass
class AlignOptions:
    enabled: bool = False
    v_mag: float = 4.5  # faintest catalogue stars used
    min_alt: float = 20.0  # deg
    max_offset: float = 2.0  # deg, largest move recovered by the grid search
    grid_step: float = 0.2  # deg
    tolerance: float = 0.15  # deg, match radius when tracking
    min_stars: int = 20
    min_fraction: float = 0.4
    snr: float = 8.0  # detection threshold
    min_sigma: float | None = None  # noise floor of the detection (see detect_stars)
    max_detections: int = 600

    @classmethod
    def from_dict(cls, options: dict[str, Any] | None) -> AlignOptions:
        options = dict(options or {})
        kwargs: dict[str, Any] = {}
        for name, value in options.items():
            if name not in cls.__dataclass_fields__:
                raise ValueError(f"Unknown align option {name}")
            if name == "enabled":
                kwargs[name] = bool(value)
            elif name in ("min_stars", "max_detections"):
                kwargs[name] = int(value)
            else:
                kwargs[name] = None if value is None else float(value)
        return cls(**kwargs)


class PointingAligner:
    def __init__(self, model: AlignedModel, observer: Observer, options: AlignOptions) -> None:
        self.model = model
        self._observer = observer
        self._o = options
        self.last_matched = 0
        self.last_fraction = 0.0

    def _px_per_deg(self, shape: tuple[int, int]) -> float:
        """Local plate scale at the image centre (pixels per degree)."""
        h, w = shape
        a1, z1 = self.model.pix_to_altaz(w / 2, h / 2)
        a2, z2 = self.model.pix_to_altaz(w / 2 + 10, h / 2)
        v1, v2 = altaz_to_enu(np.array(a1), np.array(z1)), altaz_to_enu(np.array(a2), np.array(z2))
        angle = np.rad2deg(np.arccos(np.clip(float(np.dot(v1, v2)), -1, 1)))
        return 10.0 / angle if angle > 0 else 1.0

    def _project(self, angles: FloatArray, enu: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Pixel positions of ``enu`` stars under rotation(s) ``angles`` ((3,) or (G, 3)) of the base model."""
        angles = np.atleast_2d(angles)
        q = np.stack([small_rotation(a) for a in angles])  # (G, 3, 3)
        rotated = np.einsum("gij,jn->ign", q, enu)  # (3, G, N)
        alt, az = enu_to_altaz(rotated)
        x, y = self.model.model.altaz_to_pix(alt, az)
        return np.asarray(x), np.asarray(y)

    def __call__(self, image: FloatArray, obs_time: datetime.datetime, mask: npt.NDArray[np.bool_] | None) -> bool:
        """Update the model's correction from ``image`` (in the model's pixel coordinates). True if updated."""
        o = self._o
        h, w = image.shape
        x, y, _ = detect_stars(image, snr=o.snr, max_stars=o.max_detections, mask=mask, min_sigma=o.min_sigma)
        alt, az, _ = catalog_altaz(self._observer, obs_time, o.v_mag, o.min_alt)
        if len(x) < o.min_stars or len(alt) < o.min_stars:
            return False
        enu = altaz_to_enu(alt, az)
        tree = cKDTree(np.column_stack([x, y]))
        scale = self._px_per_deg((h, w))
        tol = o.tolerance * scale

        def inside(px: FloatArray, py: FloatArray) -> npt.NDArray[np.bool_]:
            ok = np.isfinite(px) & (px > 2) & (px < w - 3) & (py > 2) & (py < h - 3)
            if mask is not None:
                ix = np.clip(np.nan_to_num(px).astype(int), 0, w - 1)
                iy = np.clip(np.nan_to_num(py).astype(int), 0, h - 1)
                ok &= mask[iy, ix]
            return np.asarray(ok)

        def score(angles: FloatArray, radius: float) -> tuple[int, int]:
            px, py = self._project(angles, enu)
            ok = inside(px[0], py[0])
            if not ok.any():
                return 0, 0
            d, _ = tree.query(np.column_stack([px[0][ok], py[0][ok]]), distance_upper_bound=radius)
            return int(np.isfinite(d).sum()), int(ok.sum())

        def refine(angles: FloatArray) -> tuple[FloatArray, int, int]:
            for radius in (2 * tol, tol, tol):
                px, py = self._project(angles, enu)
                ok = inside(px[0], py[0])
                if not ok.any():
                    return angles, 0, 0
                d, j = tree.query(np.column_stack([px[0][ok], py[0][ok]]), distance_upper_bound=radius)
                hit = np.isfinite(d)
                if hit.sum() < 6:
                    return angles, int(hit.sum()), int(ok.sum())
                sel_enu = enu[:, np.flatnonzero(ok)[hit]]
                tx, ty = x[j[hit]], y[j[hit]]

                def resid(a: FloatArray) -> FloatArray:
                    qx, qy = self._project(a, sel_enu)
                    res = np.nan_to_num(np.concatenate([qx[0] - tx, qy[0] - ty]), nan=1e3)
                    return np.asarray(res, dtype=np.float64)

                angles = optimize.least_squares(resid, angles, loss="soft_l1", f_scale=1.0).x
            n, total = score(angles, tol)
            return angles, n, total

        def good(n: int, total: int) -> bool:
            return n >= o.min_stars and total > 0 and n / total >= o.min_fraction

        current = np.array(self.model.angles)
        angles, n, total = refine(current)
        if not good(n, total):
            # grid search around the current correction
            steps = np.deg2rad(np.arange(-o.max_offset, o.max_offset + 1e-9, o.grid_step))
            ga, gb, gc = np.meshgrid(steps, steps, steps, indexing="ij")
            grid = current + np.column_stack([ga.ravel(), gb.ravel(), gc.ravel()])
            px, py = self._project(grid, enu)
            ok = inside(px.ravel(), py.ravel()).reshape(px.shape)
            pts = np.column_stack([np.where(ok, px, -1e6).ravel(), np.where(ok, py, -1e6).ravel()])
            d, _ = tree.query(pts, distance_upper_bound=max(tol, o.grid_step * scale * 0.6))
            hits = np.isfinite(d).reshape(px.shape).sum(axis=1)
            best = int(np.argmax(hits))
            angles, n, total = refine(grid[best])
        self.last_matched = n
        self.last_fraction = n / total if total else 0.0
        if not good(n, total):
            log.debug("Alignment: only %d of %d bright stars matched, keeping the correction", n, total)
            return False
        moved = np.rad2deg(np.max(np.abs(np.asarray(angles) - current)))
        self.model.set_angles(angles)
        log_fn = log.info if moved > 0.05 else log.debug
        log_fn(
            "Alignment: correction (%.3f, %.3f, %.3f) deg (changed by %.3f), %d of %d bright stars",
            *np.rad2deg(angles),
            moved,
            n,
            total,
        )
        return True
