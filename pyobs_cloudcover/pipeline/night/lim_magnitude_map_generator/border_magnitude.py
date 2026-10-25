from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.sky_query import SkyGrid, angle_to_chord, chord_to_angle

WeightFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def legacy_weight(distance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Star weight of the former Rust extension, kept for comparison only (``distance`` in rad).

    It applied the deg->rad factor to a distance already in radians and dropped the ``exp``, so the weight grew
    with distance: the farthest stars dominated and a star right at the grid point counted zero.
    """
    return -distance * np.pi / (7.0 * 180.0)


def exponential_weight(scale: float) -> WeightFunction:
    """``exp(-distance / scale)``, the intended weight (``scale`` = search radius, both in rad)."""

    def weight(distance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.exp(-distance / scale))

    return weight


def border_magnitude(
    grid: SkyGrid,
    star_alt: npt.NDArray[np.float64],
    star_az: npt.NDArray[np.float64],
    v_mag: npt.NDArray[np.float64],
    found: npt.NDArray[np.bool_],
    radius: float,
    weight: WeightFunction | None = None,
) -> npt.NDArray[np.float64]:
    """Estimate the limiting magnitude at every grid position from the seen/unseen stars within ``radius``.

    For each position, take the faintest seen star (``hv``) and the brightest unseen star (``ln``) within
    ``radius`` (rad). If they overlap (``hv > ln``), the result is the weighted mean magnitude of all stars
    between them; otherwise the weighted mean of those two border stars. With only seen (unseen) stars, the
    faintest seen (brightest unseen) magnitude is used. Positions without any star get NaN. Stars are weighted
    by ``weight(distance)``, by default ``exp(-distance / radius)``.
    """
    if weight is None:
        weight = exponential_weight(radius)
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



def robust_border_magnitude(
    grid: SkyGrid,
    star_alt: npt.NDArray[np.float64],
    star_az: npt.NDArray[np.float64],
    v_mag: npt.NDArray[np.float64],
    found: npt.NDArray[np.bool_],
    radius: float,
    weight: WeightFunction | None = None,
) -> npt.NDArray[np.float64]:
    """Limiting magnitude as the cut that best separates seen from unseen stars within ``radius``.

    For every position, the stars within ``radius`` are sorted by magnitude and the cut ``c`` is chosen that
    minimises the weighted number of stars on the wrong side of it (seen but fainter than ``c``, or unseen but
    brighter). The estimate is halfway between the stars either side of the best cut (the middle of the range
    if several cuts are equally good). Unlike :func:`border_magnitude`, which only looks at the faintest seen
    and brightest unseen star, a single false detection or missed star can't move the result far.

    Edge cases match :func:`border_magnitude`: all stars seen gives the faintest magnitude, none seen the
    brightest, no stars NaN.
    """
    if weight is None:
        weight = exponential_weight(radius)
    n = len(grid)
    out = np.full(n, np.nan)
    if n == 0 or len(v_mag) == 0:
        return out

    stars = SkyGrid(star_alt, star_az)
    pairs = grid.tree.sparse_distance_matrix(stars.tree, angle_to_chord(radius), output_type="ndarray")
    if len(pairs) == 0:
        return out
    distance = np.asarray(chord_to_angle(pairs["v"]), dtype=np.float64)
    mag_all = np.asarray(v_mag, dtype=np.float64)[pairs["j"]]
    order = np.lexsort((mag_all, pairs["i"]))
    gi = pairs["i"][order]
    mag = mag_all[order]
    vis = np.asarray(found, dtype=np.bool_)[pairs["j"]][order]
    w = weight(distance[order])

    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    groups = gi[starts]
    counts = np.diff(np.r_[starts, len(gi)])
    group_of = np.repeat(np.arange(len(starts)), counts)

    # cost of cutting after star j (j counted within its group): seen stars fainter + unseen stars up to j
    cum_seen = np.cumsum(w * vis)
    cum_unseen = np.cumsum(w * ~vis)
    before = np.r_[0.0, cum_seen][starts]
    before_unseen = np.r_[0.0, cum_unseen][starts]
    total_seen = np.add.reduceat(w * vis, starts)
    seen_upto = cum_seen - before[group_of]
    unseen_upto = cum_unseen - before_unseen[group_of]
    cost_after = (total_seen[group_of] - seen_upto) + unseen_upto
    cost_before_all = total_seen  # cut before the brightest star: every seen star is wrong

    best_after = np.minimum.reduceat(cost_after, starts)
    best = np.minimum(best_after, cost_before_all)
    eps = 1e-9 * (1 + best)

    # magnitude of each candidate cut: halfway to the next star, the last star itself at the end
    next_mag = np.r_[mag[1:], np.nan]
    is_last = np.r_[starts[1:] - 1, len(gi) - 1]
    next_mag[is_last] = mag[is_last]
    cut_mag = (mag + next_mag) / 2

    # middle of the range of optimal cuts
    optimal = cost_after <= (best + eps)[group_of]
    first_cut = np.full(len(starts), np.nan)
    last_cut = np.full(len(starts), np.nan)
    idx = np.flatnonzero(optimal)
    # cut_mag rises within a group, so min/max give the first/last optimal cut
    np.fmin.at(first_cut, group_of[idx], cut_mag[idx])
    np.fmax.at(last_cut, group_of[idx], cut_mag[idx])
    before_is_optimal = cost_before_all <= best + eps
    first_cut = np.where(before_is_optimal, mag[starts], first_cut)
    last_cut = np.where(np.isnan(last_cut), mag[starts], last_cut)

    out[groups] = (first_cut + last_cut) / 2
    return out
