"""Fit a :class:`PointedModel` (a camera that isn't a zenith-centred fisheye) from clear-night frames.

The fisheye calibration (:mod:`astrometry`) finds the zenith from the lens circle. A wide-angle camera has no lens
circle and its optical axis need not point at the zenith, so here:

1. Stars are detected as in :mod:`astrometry` (optionally inside the camera mask, to skip burnt-in labels).
2. For a grid of optical-axis directions (within ``max_tilt`` of the zenith), focal lengths, lens shapes and both
   parities, the roll about the axis is found by voting: every catalogue star / detection pair at about the same
   distance from the principal point (assumed at the image centre) votes for the roll that puts one onto the other.
3. The best candidates are refined by least squares (principal point, focal length, orientation, k1, k2), growing
   the matching radius down, and judged like the fisheye fit: the fraction of bright catalogue stars inside the
   image that are matched, against the same at wrong rolls.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from scipy import optimize

from pyobs_cloudcover.calibration.astrometry import Observation, _match, catalog_altaz, detect_stars, luminance
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.world_model.pointed_model import PointedModel

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass
class PointedResult:
    model: PointedModel
    rms: float
    matched: int
    match_fraction: float
    baseline_fraction: float
    candidates: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return (
            self.match_fraction > 3 * self.baseline_fraction
            and self.match_fraction - self.baseline_fraction > 0.2
            and self.matched >= 15
        )

    @property
    def lowest_altitude(self) -> float:
        return 0.0

    def config(self) -> dict[str, Any]:
        return self.model.to_config()


def _axis_matrix(alt: float, az: float) -> FloatArray:
    """Rows: camera x, y, z in (east, north, up) for an axis at (alt, az), roll 0 (x horizontal)."""
    z = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    x = np.array([np.cos(az), -np.sin(az), 0.0])
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _roll(m: FloatArray, roll: float) -> FloatArray:
    c, s = np.cos(roll), np.sin(roll)
    return np.asarray(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ m)


def _model_from_matrix(target: FloatArray, **lens: Any) -> PointedModel:
    """The PointedModel (rotation, tilt_x, tilt_y) whose rotation matrix is closest to ``target``."""
    best: tuple[float, FloatArray] | None = None
    for start in np.deg2rad(np.arange(0, 360, 45)):

        def resid(p: FloatArray) -> FloatArray:
            m = PointedModel(0, 0, 1.0, rotation=p[0], tilt_x=p[1], tilt_y=p[2])._matrix
            return np.asarray((m - target).ravel())

        p0 = np.array([start, 0.0, 0.0])
        res = optimize.least_squares(resid, p0)
        if best is None or res.cost < best[0]:
            best = (float(res.cost), res.x)
    assert best is not None
    rot, tx, ty = best[1]
    return PointedModel(rotation=float(np.mod(rot, 2 * np.pi)), tilt_x=float(tx), tilt_y=float(ty), **lens)


def _vote(
    obs: Sequence[Observation], m0: FloatArray, lens: PointedModel, parity: int, tol: float, bin_deg: float = 0.5
) -> FloatArray:
    nbins = int(round(360 / bin_deg))
    hist = np.zeros(nbins)
    for o in obs:
        enu = np.stack([np.cos(o.alt) * np.sin(o.az), np.cos(o.alt) * np.cos(o.az), np.sin(o.alt)])
        p = m0 @ enu
        phi = np.arccos(np.clip(p[2], -1, 1))
        ok = phi < np.deg2rad(lens.max_field_angle)
        r_cat = lens.radius(phi[ok])
        psi_cat = np.arctan2(p[1][ok], p[0][ok])
        dx, dy = o.x - lens.c_x, parity * (o.y - lens.c_y)
        r_det, psi_det = np.hypot(dx, dy), np.arctan2(dy, dx)
        order = np.argsort(r_det)
        r_sorted = r_det[order]
        lo = np.searchsorted(r_sorted, r_cat - tol)
        hi = np.searchsorted(r_sorted, r_cat + tol)
        counts = hi - lo
        if counts.sum() == 0:
            continue
        cat_idx = np.repeat(np.arange(len(r_cat)), counts)
        det_idx = order[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])]
        roll = np.mod(psi_det[det_idx] - psi_cat[cat_idx], 2 * np.pi)
        bins = (np.rad2deg(roll) / bin_deg).astype(int) % nbins
        pairs = np.unique(cat_idx * nbins + bins)
        np.add.at(hist, pairs % nbins, 1)
    return hist + np.roll(hist, 1) + np.roll(hist, -1)


def _refine(
    model: PointedModel, obs: Sequence[Observation], tol: float, shape: tuple[int, int], distortion: bool
) -> tuple[PointedModel, float, int]:
    matches = [(o, *_match(model, o, tol, shape)[:2]) for o in obs]
    alt = np.concatenate([o.alt[c] for o, c, _ in matches])
    az = np.concatenate([o.az[c] for o, c, _ in matches])
    xd = np.concatenate([o.x[d] for o, _, d in matches])
    yd = np.concatenate([o.y[d] for o, _, d in matches])
    if len(alt) < 10:
        return model, float("inf"), len(alt)
    base = model.params()

    def build(p: FloatArray) -> PointedModel:
        params = {**base, "c_x": p[0], "c_y": p[1], "f": p[2], "rotation": p[3], "tilt_x": p[4], "tilt_y": p[5]}
        if distortion:
            params.update(k1=p[6], k2=p[7])
        return PointedModel(**params)

    p0 = [model.c_x, model.c_y, model.f, model.rotation, model.tilt_x, model.tilt_y]
    if distortion:
        p0 += [model.k1, model.k2]

    def residuals(p: FloatArray) -> FloatArray:
        try:
            m = build(p)
        except ValueError:
            return np.full(2 * len(alt), 1e3)
        x, y = (np.asarray(v) for v in m.altaz_to_pix(alt, az))
        return np.asarray(np.nan_to_num(np.concatenate([x - xd, y - yd]), nan=1e3), dtype=np.float64)

    res = optimize.least_squares(residuals, np.array(p0), loss="soft_l1", f_scale=1.0, x_scale="jac")
    try:
        fitted = build(res.x)
    except ValueError:
        return model, float("inf"), len(alt)
    final = residuals(res.x).reshape(2, -1)
    rms = float(np.sqrt(np.median(np.sum(final**2, axis=0))))
    return fitted, rms, len(alt)


def _fractions(
    model: PointedModel, obs: Sequence[Observation], tol: float, shape: tuple[int, int]
) -> tuple[float, float]:
    def fraction(m: PointedModel) -> float:
        hits, total = 0, 0
        for o in obs:
            c, _, inside = _match(m, o, tol, shape)
            hits, total = hits + len(c), total + inside
        return hits / total if total else 0.0

    wrong = []
    for offset in (20, 45, 90, 135, 180, 225, 270, 315, 340):
        target = _roll(model._matrix, np.deg2rad(offset))
        lens = {k: v for k, v in model.params().items() if k not in ("rotation", "tilt_x", "tilt_y")}
        wrong.append(fraction(_model_from_matrix(target, **lens)))
    return fraction(model), float(np.median(wrong))


def calibrate_pointed(
    frames: Sequence[Frame],
    observer: Observer,
    hfov: float = 100.0,
    max_tilt: float = 20.0,
    focal_range: tuple[float, float] = (0.8, 1.25),
    match_mag: float = 4.5,
    vote_mag: float = 3.0,
    mask: npt.NDArray[np.bool_] | None = None,
    max_field_angle: float = 80.0,
    min_sigma: float | None = None,
) -> PointedResult:
    """Fit a :class:`PointedModel`. ``hfov``: rough horizontal field of view (deg), sets the focal-length search
    range together with ``focal_range``. ``max_tilt``: how far (deg) the optical axis may be from the zenith."""
    if not frames:
        raise ValueError("No frames")
    images = [luminance(f) for f in frames]
    shape = images[0].shape
    h, w = shape
    obs: list[Observation] = []
    for frame, image in zip(frames, images):
        x, y, _ = detect_stars(image, mask=mask, max_stars=1500, min_sigma=min_sigma)
        alt, az, v = catalog_altaz(observer, frame.obs_time, max(match_mag, vote_mag), min_alt=5.0)
        obs.append(Observation(x, y, alt, az, v))
        log.info("%s: %d detections, %d catalogue stars", frame.path or frame.obs_time, len(x), len(alt))

    # vote with the bright stars and the brightest detections
    vote_obs = []
    for o in obs:
        bright = o.v_mag <= vote_mag
        vote_obs.append(Observation(o.x[:300], o.y[:300], o.alt[bright], o.az[bright], o.v_mag[bright]))

    half = np.deg2rad(hfov / 2)
    c_x, c_y = (w - 1) / 2, (h - 1) / 2
    candidates: list[tuple[float, ...]] = []
    shapes = {"rectilinear": (1 / 3, 2 / 15), "between": (1 / 6, 0.0), "equidistant": (0.0, 0.0)}
    for name, (k1, k2) in shapes.items():
        lens0 = PointedModel(0, 0, 1.0, k1=k1, k2=k2, max_field_angle=max_field_angle)
        f0 = (w / 2) / float(lens0.radius(np.array(half)))
        for scale in np.arange(focal_range[0], focal_range[1] + 1e-9, 0.05):
            f = f0 * scale
            lens = PointedModel(c_x, c_y, f, k1=k1, k2=k2, max_field_angle=max_field_angle)
            tol = f * np.deg2rad(1.0)
            for tilt in np.arange(0.0, max_tilt + 1e-9, 2.0):
                n_az = 1 if tilt == 0 else max(4, int(np.ceil(360 * np.sin(np.deg2rad(tilt)) / 2.0)))
                for axis_az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
                    m0 = _axis_matrix(np.deg2rad(90 - tilt), axis_az)
                    for parity in (1, -1):
                        hist = _vote(vote_obs, m0, lens, parity, tol)
                        best = int(np.argmax(hist))
                        roll = np.deg2rad((best + 0.5) * 0.5)
                        candidates.append((hist[best], k1, k2, f, tilt, axis_az, roll, parity))
    candidates.sort(key=lambda c: -c[0])
    log.info(
        "Best vote candidates (votes, f, tilt, axis az, roll, parity): %s",
        [(int(c[0]), round(c[3]), c[4], round(np.rad2deg(c[5])), round(np.rad2deg(c[6])), c[7])
         for c in candidates[:5]],
    )

    best_result: PointedResult | None = None
    seen: list[PointedModel] = []
    for votes, c_k1, c_k2, c_f, c_tilt, c_az, c_roll, c_parity in candidates[:12]:
        target = _roll(_axis_matrix(np.deg2rad(90 - c_tilt), c_az), c_roll)
        model = _model_from_matrix(
            target, c_x=c_x, c_y=c_y, f=c_f, k1=c_k1, k2=c_k2, parity=int(c_parity), max_field_angle=max_field_angle
        )
        if any(_same(model, s) for s in seen):
            continue
        seen.append(model)
        tol0 = c_f * np.deg2rad(1.0)
        tol = tol0
        rms = float("inf")
        n = 0
        for step in range(8):
            model, rms, n = _refine(model, obs, tol, shape, distortion=step >= 2)
            if not np.isfinite(rms):
                break
            tol = max(2.0, min(tol0, 4 * rms))
        if not np.isfinite(rms):
            continue
        fraction, baseline = _fractions(model, vote_obs, max(3.0, 3 * rms), shape)
        result = PointedResult(model, rms, n, fraction, baseline, candidates[:10])
        log.info("candidate %d votes -> rms %.2f px, %d stars, fraction %.2f (baseline %.2f)", votes, rms, n,
                 fraction, baseline)
        if best_result is None or (result.match_fraction, -result.rms) > (
            best_result.match_fraction,
            -best_result.rms,
        ):
            best_result = result
    if best_result is None:
        raise ValueError("No candidate could be refined; are the frames clear and the timestamps right?")
    best_alt, best_az = best_result.model.axis_altaz
    log.info(
        "Best: axis alt %.2f az %.2f deg, f %.1f px, k1 %.3f k2 %.3f, rms %.2f px, %d stars, match fraction %.2f "
        "(wrong rolls %.2f) -> %s",
        np.rad2deg(best_alt), np.rad2deg(best_az), best_result.model.f, best_result.model.k1, best_result.model.k2,
        best_result.rms, best_result.matched, best_result.match_fraction, best_result.baseline_fraction,
        "trustworthy" if best_result.trustworthy else "NOT trustworthy",
    )
    return best_result


def _same(a: PointedModel, b: PointedModel) -> bool:
    return a.parity == b.parity and bool(np.allclose(a._matrix, b._matrix, atol=np.deg2rad(2))) and abs(
        a.f / b.f - 1) < 0.03
