from typing import Callable

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.sky_query import SkyGrid, angle_to_chord, chord_to_angle

WeightFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def legacy_weight(distance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Star weight as implemented in the former Rust extension (``distance`` in rad)."""
    return -distance * np.pi / (7.0 * 180.0)


def border_magnitude(
    grid: SkyGrid,
    star_alt: npt.NDArray[np.float64],
    star_az: npt.NDArray[np.float64],
    v_mag: npt.NDArray[np.float64],
    found: npt.NDArray[np.bool_],
    radius: float,
    weight: WeightFunction = legacy_weight,
) -> npt.NDArray[np.float64]:
    """Estimate the limiting magnitude at every grid position from the seen/unseen stars within ``radius``.

    For each position, take the faintest seen star (``hv``) and the brightest unseen star (``ln``) within
    ``radius`` (rad). If they overlap (``hv > ln``), the result is the weighted mean magnitude of all stars
    between them; otherwise the weighted mean of those two border stars. With only seen (unseen) stars, the
    faintest seen (brightest unseen) magnitude is used. Positions without any star get NaN.
    """
    n = len(grid)
    out = np.full(n, np.nan)
    if n == 0 or len(v_mag) == 0:
        return out

    stars = SkyGrid(star_alt, star_az)
    pairs = grid.tree.sparse_distance_matrix(stars.tree, angle_to_chord(radius), output_type="ndarray")
    gi, si = pairs["i"], pairs["j"]
    distance = np.asarray(chord_to_angle(pairs["v"]), dtype=np.float64)

    mag = np.asarray(v_mag, dtype=np.float64)[si]
    vis = np.asarray(found, dtype=np.bool_)[si]
    w = weight(distance)

    # border stars: faintest seen and brightest unseen star per position; on ties the nearest one
    seen, unseen = np.flatnonzero(vis), np.flatnonzero(~vis)
    hv_idx = _pick(gi, seen, np.lexsort((-distance[seen], mag[seen], gi[seen])), last=True, n=n)
    ln_idx = _pick(gi, unseen, np.lexsort((distance[unseen], mag[unseen], gi[unseen])), last=False, n=n)
    has_seen, has_unseen = hv_idx >= 0, ln_idx >= 0
    hv_idx = np.where(has_seen, hv_idx, ln_idx)
    ln_idx = np.where(has_unseen, ln_idx, hv_idx)
    has_star = has_seen | has_unseen

    hv = np.full(n, np.nan)
    ln = np.full(n, np.nan)
    hv[has_star] = mag[hv_idx[has_star]]
    ln[has_star] = mag[ln_idx[has_star]]

    # no overlap: weighted mean of the two border stars
    with np.errstate(invalid="ignore", divide="ignore"):
        w_hv, w_ln = w[hv_idx[has_star]], w[ln_idx[has_star]]
        out[has_star] = (hv[has_star] * w_hv + ln[has_star] * w_ln) / (w_hv + w_ln)

    # overlap: weighted mean of all stars between the brightest unseen and the faintest seen star
    overlap = has_star & (hv > ln)
    selected = overlap[gi] & (mag >= ln[gi]) & (mag <= hv[gi])
    weight_sum = np.bincount(gi[selected], w[selected], minlength=n)
    weighted_mag = np.bincount(gi[selected], (w * mag)[selected], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[overlap] = weighted_mag[overlap] / weight_sum[overlap]
    return out


def _pick(
    group: npt.NDArray[np.intp], members: npt.NDArray[np.intp], order: npt.NDArray[np.intp], last: bool, n: int
) -> npt.NDArray[np.intp]:
    """Per group (grid position), the first or last member in ``order``; -1 for groups without members."""
    result = np.full(n, -1, dtype=np.intp)
    if len(members) == 0:
        return result
    ordered = members[order]
    g = group[ordered]
    boundary = g[1:] != g[:-1]
    take = np.r_[boundary, True] if last else np.r_[True, boundary]
    result[g[take]] = ordered[take]
    return result

