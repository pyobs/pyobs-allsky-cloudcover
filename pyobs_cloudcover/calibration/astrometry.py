"""Fit a world model (lens, orientation) from clear-night frames, with no initial guess but the site and time.

1. The lens circle (the illuminated disc) gives the zenith position and a rough scale.
2. Stars are detected (local maxima above a robust local background).
3. For every projection, mirror parity and a range of focal lengths, the azimuth rotation is found by voting:
   every pair of a catalogue star and a detection at about the same distance from the centre votes for the
   rotation that would put one onto the other. The right solution collects far more votes than any other.
4. Catalogue stars are matched to detections and the model is refined by least squares (centre, rotation,
   focal length, optionally a polynomial lens). The fit is only trusted if it matches clearly more stars than
   wrong rotations do (the match fraction against a baseline, not the peak height, which glare can inflate).
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from astroplan import Observer
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from scipy import ndimage, optimize
from scipy.spatial import cKDTree

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.data import bundled_catalog_path
from pyobs_cloudcover.world_model.fisheye_model import FisheyeModel

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
PROJECTIONS = ("equidistant", "equisolid", "stereographic", "orthographic")


# ------------------------------------------------------------------------------------------------ image helpers


# a 3x3 binomial kernel weights R:G:B as 1:2:1 at every pixel of any Bayer mosaic, without shifting the image
_BAYER_KERNEL = np.outer([1, 2, 1], [1, 2, 1]) / 16.0


def luminance(frame: Frame) -> FloatArray:
    """Full-resolution luminance: RGB averaged; Bayer mosaics smoothed so the pattern vanishes."""
    data = np.asarray(frame.data, dtype=np.float64)
    if data.ndim == 3:
        return np.asarray(data.mean(axis=2))
    if frame.bayer is not None:
        return np.asarray(ndimage.convolve(data, _BAYER_KERNEL, mode="nearest"))
    return data


def _block_stats(image: FloatArray, block: int) -> tuple[FloatArray, FloatArray]:
    """Median and robust sigma in blocks, interpolated back to full size."""
    h, w = image.shape
    nh, nw = max(1, h // block), max(1, w // block)
    cut = image[: nh * block, : nw * block].reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(nh, nw, -1)
    med = np.median(cut, axis=2)
    sig = 1.4826 * np.median(np.abs(cut - med[..., None]), axis=2)
    zoom = (h / nh, w / nw)
    return (
        np.asarray(ndimage.zoom(med, zoom, order=1, mode="nearest")[:h, :w]),
        np.asarray(ndimage.zoom(sig, zoom, order=1, mode="nearest")[:h, :w]),
    )


@dataclass
class LensCircle:
    c_x: float
    c_y: float
    radius: float
    rms: float


def fit_lens_circle(images: Sequence[FloatArray]) -> LensCircle:
    """Centre and radius of the illuminated disc, fitted to its edge.

    Two discs are tried: everything lit above the noise of the dark border (in daylight that is the whole image
    circle, ground and trees included) and the bright sky alone; the one with the more circular edge wins. Edge
    points on the image border, where the circle is cut off, are ignored, and the fit follows the outer envelope
    of the edge, since obstructions only cut into the disc.
    """
    mean = np.mean([ndimage.uniform_filter(np.asarray(im, dtype=np.float64), 5) for im in images], axis=0)
    lo = float(np.percentile(mean, 2))
    dark = mean[mean <= np.percentile(mean, 10)]
    noise = 1.4826 * float(np.median(np.abs(dark - np.median(dark)))) + 1e-9
    sky = float(np.median(mean[mean > np.percentile(mean, 50)]))
    best: LensCircle | None = None
    for threshold in (lo + 8 * noise, lo + 0.5 * (sky - lo)):
        try:
            circle = _circle_from_disc(mean > threshold)
        except ValueError:
            continue
        if best is None or circle.rms < best.rms:
            best = circle
    if best is None:
        raise ValueError("No illuminated disc found")
    return best


def _circle_from_disc(inside: npt.NDArray[np.bool_]) -> LensCircle:
    labels, n = ndimage.label(inside)
    if n == 0:
        raise ValueError("No illuminated disc found")
    sizes = ndimage.sum(inside, labels, range(1, n + 1))
    disc = labels == (int(np.argmax(sizes)) + 1)
    disc = ndimage.binary_fill_holes(disc)
    edge = disc & ~ndimage.binary_erosion(disc)
    h, w = disc.shape
    ys, xs = np.nonzero(edge)
    keep = (xs > 1) & (xs < w - 2) & (ys > 1) & (ys < h - 2)
    xs, ys = xs[keep].astype(float), ys[keep].astype(float)
    if len(xs) < 20:
        raise ValueError("Lens circle edge not found")
    # Obstructions (trees, buildings) and the horizon cut *into* the disc, so fit its outer envelope: per angular
    # sector keep the outermost edge point, then iteratively drop points well inside the fitted circle.
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    for _ in range(4):
        angle = np.arctan2(ys - cy, xs - cx)
        radius = np.hypot(xs - cx, ys - cy)
        sector = ((angle + np.pi) / (2 * np.pi) * 180).astype(int) % 180
        outer = np.zeros(len(xs), bool)
        for k in np.unique(sector):
            idx = np.flatnonzero(sector == k)
            outer[idx[np.argmax(radius[idx])]] = True
        px, py = xs[outer], ys[outer]
        cx, cy, r = _fit_circle(px, py)
        for _ in range(5):
            residual = np.hypot(px - cx, py - cy) - r
            spread = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 0.5
            keep = residual > -2.5 * spread
            if keep.all() or keep.sum() < 10:
                break
            px, py = px[keep], py[keep]
            cx, cy, r = _fit_circle(px, py)
    rms = float(np.sqrt(np.mean((np.hypot(px - cx, py - cy) - r) ** 2)))
    return LensCircle(float(cx), float(cy), float(r), rms)


def _fit_circle(xs: FloatArray, ys: FloatArray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit, refined geometrically."""
    a = np.column_stack([xs, ys, np.ones_like(xs)])
    c = np.linalg.lstsq(a, xs**2 + ys**2, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx**2 + cy**2)
    res = optimize.least_squares(lambda p: np.hypot(xs - p[0], ys - p[1]) - p[2], [cx, cy, r], loss="soft_l1")
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def detect_stars(
    image: FloatArray, snr: float = 6.0, max_stars: int = 400, mask: npt.NDArray[np.bool_] | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Positions (x, y) and peak SNR of the brightest point sources: local maxima above a robust local background,
    with sub-pixel centroids."""
    background, sigma = _block_stats(image, 32)
    sigma = np.maximum(sigma, 1e-6)
    significance = (ndimage.gaussian_filter(image, 1.0) - background) / sigma
    peaks = (significance == ndimage.maximum_filter(significance, size=7)) & (significance > snr)
    if mask is not None:
        peaks &= mask
    ys, xs = np.nonzero(peaks)
    order = np.argsort(significance[ys, xs])[::-1][:max_stars]
    ys, xs = ys[order], xs[order]
    # 3x3 centroid
    h, w = image.shape
    cx, cy = xs.astype(float), ys.astype(float)
    total = np.zeros(len(xs))
    sx, sy = np.zeros(len(xs)), np.zeros(len(xs))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
            v = np.clip(image[yy, xx] - background[yy, xx], 0, None)
            total += v
            sx += v * dx
            sy += v * dy
    ok = total > 0
    cx[ok] += sx[ok] / total[ok]
    cy[ok] += sy[ok] / total[ok]
    return cx, cy, significance[ys, xs]


# ------------------------------------------------------------------------------------------------ catalogue


def catalog_altaz(
    observer: Observer, time: datetime.datetime, max_mag: float, min_alt: float = 10.0, catalog: str | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    data = np.loadtxt(catalog or bundled_catalog_path(), skiprows=1, delimiter=";", comments=["#"])
    bright = data[:, 4] <= max_mag
    coords = SkyCoord(ra=data[bright, 2], dec=data[bright, 3], unit="deg")
    altaz = coords.transform_to(AltAz(obstime=Time(time), location=observer.location))
    alt, az = altaz.alt.rad, altaz.az.rad
    above = alt > np.deg2rad(min_alt)
    return alt[above], az[above], data[bright, 4][above]


# ------------------------------------------------------------------------------------------------ fit


@dataclass
class Observation:
    """One frame's detections and catalogue."""

    x: FloatArray
    y: FloatArray
    alt: FloatArray
    az: FloatArray
    v_mag: FloatArray


@dataclass
class AstrometryResult:
    model: FisheyeModel
    rms: float  # pixels
    matched: int
    match_fraction: float  # of the bright (vote_mag) catalogue stars inside the image
    baseline_fraction: float  # the same at wrong rotations
    circle: LensCircle | None = None
    candidates: list[tuple[str, int, float, float, int]] = field(default_factory=list)
    lowest_altitude: float = 0.0  # deg, lowest matched star: the model is extrapolated below

    @property
    def trustworthy(self) -> bool:
        """Clearly more matches than at wrong rotations, and a sensible residual."""
        return (
            self.match_fraction > 3 * self.baseline_fraction
            and self.match_fraction - self.baseline_fraction > 0.2
            and self.matched >= 15
        )

    def config(self) -> dict[str, Any]:
        return self.model.to_config()


def _radius_for(projection: str, f: float, theta: FloatArray) -> FloatArray:
    return FisheyeModel(0, 0, f, projection=projection).radius(theta)


def _vote(
    obs: list[Observation], projection: str, parity: int, f: float, cx: float, cy: float, tol_r: float, bin_deg: float
) -> FloatArray:
    """Histogram of azimuth rotations implied by catalogue/detection pairs at matching radii."""
    nbins = int(round(360 / bin_deg))
    hist = np.zeros(nbins)
    for o in obs:
        r_cat = _radius_for(projection, f, np.pi / 2 - o.alt)
        dx, dy = o.x - cx, o.y - cy
        r_det = np.hypot(dx, dy)
        phi_det = np.arctan2(dy, dx)
        order = np.argsort(r_det)
        r_sorted = r_det[order]
        lo = np.searchsorted(r_sorted, r_cat - tol_r)
        hi = np.searchsorted(r_sorted, r_cat + tol_r)
        counts = hi - lo
        if counts.sum() == 0:
            continue
        cat_idx = np.repeat(np.arange(len(r_cat)), counts)
        det_idx = order[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])]
        # x = cx + r cos(parity * (az - a0))  =>  a0 = az - parity * phi
        a0 = np.mod(o.az[cat_idx] - parity * phi_det[det_idx], 2 * np.pi)
        # each catalogue star votes at most once per bin
        bins = (np.rad2deg(a0) / bin_deg).astype(int) % nbins
        pairs = np.unique(cat_idx * nbins + bins)
        np.add.at(hist, pairs % nbins, 1)
    # smooth over neighbouring bins, a rotation between bins should still score
    return hist + np.roll(hist, 1) + np.roll(hist, -1)


def _match(
    model: FisheyeModel, o: Observation, tol: float, shape: tuple[int, int]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """Catalogue stars (indices) matched to detections (indices) within ``tol`` px, and how many catalogue stars
    fell inside the image."""
    x, y = (np.asarray(v) for v in model.altaz_to_pix(o.alt, o.az))
    h, w = shape
    inside = np.isfinite(x) & (x > 2) & (x < w - 3) & (y > 2) & (y < h - 3)
    if not np.any(inside) or len(o.x) == 0:
        return np.array([], np.intp), np.array([], np.intp), int(inside.sum())
    tree = cKDTree(np.column_stack([o.x, o.y]))
    dist, j = tree.query(np.column_stack([x[inside], y[inside]]), distance_upper_bound=tol)
    ok = np.isfinite(dist)
    cat = np.flatnonzero(inside)[ok]
    det = j[ok]
    # one catalogue star per detection: keep the closest
    order = np.argsort(dist[ok])
    _, first = np.unique(det[order], return_index=True)
    keep = order[first]
    return cat[keep], det[keep], int(inside.sum())


def _above(o: Observation, min_alt: float) -> Observation:
    keep = o.alt >= np.deg2rad(min_alt)
    return Observation(o.x, o.y, o.alt[keep], o.az[keep], o.v_mag[keep])


def _refine(
    model: FisheyeModel,
    obs: list[Observation],
    tol: float,
    shape: tuple[int, int],
    polynomial: bool,
    min_alt: float = 0.0,
) -> tuple[FisheyeModel, float, int]:
    """Least-squares fit of centre, rotation and radial function to the stars above ``min_alt`` (deg) matched
    within ``tol`` pixels."""
    matches = [(o, *_match(model, o, tol, shape)[:2]) for o in (_above(o, min_alt) for o in obs)]
    alt = np.concatenate([o.alt[c] for o, c, _ in matches])
    az = np.concatenate([o.az[c] for o, c, _ in matches])
    xd = np.concatenate([o.x[d] for o, _, d in matches])
    yd = np.concatenate([o.y[d] for o, _, d in matches])
    if len(alt) < 6:
        return model, float("inf"), len(alt)
    theta = np.pi / 2 - alt

    if polynomial:
        # Only trust the polynomial a little beyond the lowest matched star, and only use a 5th-order term when
        # stars reach low altitudes; otherwise it extrapolates wildly.
        lowest = float(np.rad2deg(alt.min()))
        # cover at least the altitude the fit is growing to, or stars there could never be matched
        max_zenith = min(model.max_zenith_angle, 90 - min(lowest, min_alt) + 10)
        powers = (1, 3, 5) if lowest < 25 else (1, 3)
        t = np.linspace(0, np.deg2rad(max_zenith), 200)
        coeffs0 = np.linalg.lstsq(np.column_stack([t**k for k in powers]), model.radius(t), rcond=None)[0]

        def build(p: FloatArray) -> FisheyeModel:
            coefficients = [0.0] * max(powers)
            for k, c in zip(powers, p[3:]):
                coefficients[k - 1] = float(c)
            return FisheyeModel(
                p[0],
                p[1],
                a0=p[2],
                projection="polynomial",
                coefficients=coefficients,
                parity=model.parity,
                max_zenith_angle=max_zenith,
            )

        p0 = np.r_[model.c_x, model.c_y, model.a0, coeffs0]
    else:

        def build(p: FloatArray) -> FisheyeModel:
            return FisheyeModel(
                p[0],
                p[1],
                p[3],
                a0=p[2],
                projection=model.projection,
                parity=model.parity,
                max_zenith_angle=model.max_zenith_angle,
            )

        p0 = np.r_[model.c_x, model.c_y, model.a0, model.f]

    def residuals(p: FloatArray) -> FloatArray:
        m = build(p)
        r = np.asarray(m.radius(theta))
        phi = m.parity * (az - p[2])
        return np.concatenate([p[0] + r * np.cos(phi) - xd, p[1] + r * np.sin(phi) - yd])

    try:
        res = optimize.least_squares(residuals, p0, loss="soft_l1", f_scale=1.0)
        fitted = build(res.x)
    except ValueError:
        return model, float("inf"), len(alt)
    final = residuals(res.x).reshape(2, -1)
    rms = float(np.sqrt(np.mean(np.sum(final**2, axis=0))))
    return fitted, rms, len(alt)


def calibrate(
    frames: Sequence[Frame],
    observer: Observer,
    match_mag: float = 4.5,
    vote_mag: float = 3.5,
    focal_range: tuple[float, float] = (0.8, 1.25),
    max_zenith_angle: float = 100.0,
    polynomial: bool | None = None,
    circle_frames: Sequence[Frame] | None = None,
) -> AstrometryResult:
    """Fit a :class:`FisheyeModel` to clear-night frames (canonical orientation, see :class:`CameraProfile`).

    ``polynomial``: also fit a polynomial lens and keep it if it fits clearly better (None), always (True) or
    never (False). ``circle_frames``: images to find the lens circle in instead of the night frames, e.g.
    daytime frames, where the whole image circle is lit, when trees or buildings hide much of the horizon; the frames
    giving the more circular edge are used.
    """
    if not frames:
        raise ValueError("No frames")
    images = [luminance(f) for f in frames]
    shape = images[0].shape
    circle = fit_lens_circle(images)
    if circle_frames:
        # whichever frames show the cleaner circle: daytime frames help when obstructions hide the horizon at
        # night, but outside some lenses it isn't dark by day (scattered light)
        try:
            day_circle = fit_lens_circle([luminance(f) for f in circle_frames])
        except ValueError:
            day_circle = None
        if day_circle is not None and day_circle.rms < circle.rms:
            circle = day_circle
    log.info(
        "Lens circle: centre (%.1f, %.1f), radius %.1f px (edge rms %.2f px)",
        circle.c_x,
        circle.c_y,
        circle.radius,
        circle.rms,
    )

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    in_circle = np.hypot(xx - circle.c_x, yy - circle.c_y) < 0.97 * circle.radius
    obs = []
    for frame, image in zip(frames, images):
        x, y, _ = detect_stars(image, mask=in_circle, max_stars=1500)
        alt, az, v = catalog_altaz(observer, frame.obs_time, max(match_mag, vote_mag))
        obs.append(Observation(x, y, alt, az, v))
        log.info("%s: %d detections, %d catalogue stars", frame.path or frame.obs_time, len(x), len(alt))

    vote_obs = [
        Observation(o.x, o.y, o.alt[o.v_mag <= vote_mag], o.az[o.v_mag <= vote_mag], o.v_mag[o.v_mag <= vote_mag])
        for o in obs
    ]
    # plate scale: tolerance of ~0.4 deg in radius and position
    candidates: list[tuple[str, int, float, float, int]] = []
    for projection in PROJECTIONS:
        theta_edge = np.deg2rad(min(max_zenith_angle, 90.0 if projection == "orthographic" else max_zenith_angle))
        f0 = circle.radius / float(_radius_for(projection, 1.0, np.array(theta_edge)))
        for scale in np.arange(focal_range[0], focal_range[1] + 1e-9, 0.01):
            f = f0 * scale
            tol = max(1.5, f * np.deg2rad(0.4))
            for parity in (1, -1):
                hist = _vote(vote_obs, projection, parity, f, circle.c_x, circle.c_y, tol, 0.5)
                best = int(np.argmax(hist))
                candidates.append((projection, parity, f, np.deg2rad((best + 0.5) * 0.5), int(hist[best])))
    candidates.sort(key=lambda c: -c[4])

    best_result: AstrometryResult | None = None
    for projection, parity, f, a0, votes in candidates[:6]:
        model = FisheyeModel(
            circle.c_x, circle.c_y, f, a0=a0, projection=projection, parity=parity, max_zenith_angle=max_zenith_angle
        )
        tol0 = max(2.0, f * np.deg2rad(0.6))
        tol = tol0
        # grow the fit from the zenith outwards, so a wrong start can't exclude the low stars for good
        for min_alt in (60, 45, 30, 20, 10):
            model, rms, n = _refine(model, obs, tol, shape, polynomial=False, min_alt=min_alt)
            tol = max(1.5, min(tol0, 4 * rms))
        if polynomial is not False:
            # start from the named fit, which already reaches low altitudes, on all stars with the generous
            # tolerance, then tighten (growing outwards again would let the polynomial's extrapolation lose them)
            poly, ptol = model, tol0
            for _ in range(4):
                poly, prms, pn = _refine(poly, obs, ptol, shape, polynomial=True, min_alt=10)
                ptol = max(1.5, min(tol0, 4 * prms))
            # judge both on the whole sky at the same, generous tolerance
            n_named = sum(len(_match(model, o, tol0, shape)[0]) for o in obs)
            n_poly = sum(len(_match(poly, o, tol0, shape)[0]) for o in obs)
            if polynomial or n_poly > 1.03 * n_named or (n_poly >= n_named and prms < 0.9 * rms):
                model, rms, n, tol = poly, prms, pn, ptol
        fraction, baseline = _match_fractions(model, vote_obs, tol0, shape)
        lowest = _lowest_reliable_altitude(model, vote_obs, max(1.5, 2 * rms), shape)
        result = AstrometryResult(model, rms, n, fraction, baseline, circle, candidates[:10], lowest)
        if best_result is None or (result.match_fraction, -result.rms) > (best_result.match_fraction, -best_result.rms):
            best_result = result
    assert best_result is not None
    log.info(
        "Best: %s, rms %.2f px, %d stars, match fraction %.2f (wrong rotations: %.2f) -> %s",
        best_result.model.projection,
        best_result.rms,
        best_result.matched,
        best_result.match_fraction,
        best_result.baseline_fraction,
        "trustworthy" if best_result.trustworthy else "NOT trustworthy",
    )
    return best_result


def _lowest_reliable_altitude(
    model: FisheyeModel, obs: list[Observation], tol: float, shape: tuple[int, int], min_fraction: float = 0.3
) -> float:
    """Lowest altitude (deg) down to which, in every 5 deg band, at least ``min_fraction`` of the bright
    catalogue stars in the image are matched; below it the model is unconstrained (extrapolated)."""
    alt_all, hit = [], []
    for o in obs:
        cat, _, _ = _match(model, o, tol, shape)
        x, y = (np.asarray(v) for v in model.altaz_to_pix(o.alt, o.az))
        h, w = shape
        inside = np.isfinite(x) & (x > 2) & (x < w - 3) & (y > 2) & (y < h - 3)
        matched = np.zeros(len(o.alt), bool)
        matched[cat] = True
        alt_all.append(np.rad2deg(o.alt[inside]))
        hit.append(matched[inside])
    alt, ok = np.concatenate(alt_all), np.concatenate(hit)
    lowest = 90.0
    for top in np.arange(90, 0, -5):
        band = (alt < top) & (alt >= top - 5)
        if band.sum() < 3:
            continue  # too few stars to tell (e.g. near the zenith)
        if np.mean(ok[band]) < min_fraction:
            break
        lowest = float(top - 5)
    return lowest


def _match_fractions(
    model: FisheyeModel, obs: list[Observation], tol: float, shape: tuple[int, int]
) -> tuple[float, float]:
    def fraction(m: FisheyeModel) -> float:
        hits, total = 0, 0
        for o in obs:
            c, _, inside = _match(m, o, tol, shape)
            hits, total = hits + len(c), total + inside
        return hits / total if total else 0.0

    config = model.to_config()
    config.pop("class")
    wrong = []
    for offset in (20, 45, 90, 135, 180, 225, 270, 315, 340):
        wrong.append(fraction(FisheyeModel(**{**config, "a0": model.a0 + np.deg2rad(offset)})))
    return fraction(model), float(np.median(wrong))
