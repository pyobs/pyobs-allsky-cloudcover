"""Benchmark: legacy Rust extension (cloudmap_rs) vs. a NumPy/SciPy re-implementation.

Phase 0 evidence for the Rust-vs-Python decision (see PLAN.md). Needs the legacy extension built
(``maturin develop --release`` on the commit that still has ``src/``) plus scipy.

Two workloads are timed:

* **night**: limiting-magnitude map: for every sky grid point, the border magnitude between
  seen and unseen catalogue stars within ``radius`` (``MagnitudeMapGenerator.query_many``).
* **query**: ``SkyPixelQuery`` build + the radius queries the measurement log runs per frame
  (total, zenith, moon, sun) + point queries, at night-grid and full-resolution day-pixel sizes.

Run:  python benchmarks/rust_vs_numpy.py [--threads N]
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Alt/az (rad) of n points equidistributed on the sphere."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    az = np.mod(np.pi * (1 + 5**0.5) * i, 2 * np.pi)
    return np.arcsin(z), az


def to_xyz(alt: np.ndarray, az: np.ndarray) -> np.ndarray:
    return np.column_stack([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])


def chord(angle: float) -> float:
    return 2 * np.sin(angle / 2)


def random_stars(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n stars above 30 deg, BSC-like magnitude distribution, ~70% seen (partly cloudy)."""
    alt = np.arcsin(rng.uniform(np.sin(np.deg2rad(30)), 1, n))
    az = rng.uniform(0, 2 * np.pi, n)
    vmag = np.round(6.5 - rng.exponential(1.0, n).clip(0, 7.5), 2)
    found = rng.random(n) < 0.7
    return alt, az, vmag, found


def lim_mag_numpy(
    grid_alt: np.ndarray,
    grid_az: np.ndarray,
    star_alt: np.ndarray,
    star_az: np.ndarray,
    vmag: np.ndarray,
    found: np.ndarray,
    radius: float,
    weight: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Vectorised border-magnitude estimate, same rules as src/night/star_counter.rs."""
    tree = cKDTree(to_xyz(star_alt, star_az))
    grid = cKDTree(to_xyz(grid_alt, grid_az))
    pairs = grid.sparse_distance_matrix(tree, chord(radius), output_type="ndarray")
    gi, si = pairs["i"], pairs["j"]
    dist = 2 * np.arcsin(np.clip(pairs["v"] / 2, 0, 1))
    n = len(grid_alt)

    m = vmag[si]
    w = weight(dist)
    vis = found[si]

    hv = np.full(n, -np.inf)
    np.maximum.at(hv, gi[vis], m[vis])
    ln = np.full(n, np.inf)
    np.minimum.at(ln, gi[~vis], m[~vis])
    has_v, has_n = np.isfinite(hv), np.isfinite(ln)
    hv = np.where(has_v, hv, ln)
    ln = np.where(has_n, ln, hv)

    overlap = hv > ln
    # overlap: every star with ln <= m <= hv; otherwise: the stars at hv (seen) and ln (unseen)
    sel = np.where(
        overlap[gi],
        (m >= ln[gi]) & (m <= hv[gi]),
        np.where(vis, m == hv[gi], m == ln[gi]),
    )
    sw = np.bincount(gi[sel], w[sel], minlength=n)
    swm = np.bincount(gi[sel], (w * m)[sel], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = swm / sw
    out[~(has_v | has_n)] = np.nan
    return out


def timeit(fn: Callable[[], Any], repeat: int = 3) -> tuple[float, Any]:
    best, res = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def bench_night(rust: Any, grid_n: int, n_stars: int, rng: np.random.Generator) -> None:
    galt, gaz = fibonacci_sphere(grid_n)
    keep = galt >= np.deg2rad(30)
    galt, gaz = galt[keep], gaz[keep]
    salt, saz, vmag, found = random_stars(n_stars, rng)
    radius = np.deg2rad(7.0)

    # legacy weight, as in src/night/cloud_map_generator.rs
    legacy_w = lambda d: -d * np.pi / (7.0 * 180.0)  # noqa: E731

    t_np, res_np = timeit(lambda: lim_mag_numpy(galt, gaz, salt, saz, vmag, found, radius, legacy_w))

    line = f"night grid={len(galt):>7d} stars={n_stars:>6d}  numpy {t_np * 1e3:8.1f} ms"
    if rust is not None:

        def run_rust() -> np.ndarray:
            coords = [rust.AltAzCoord(a, z) for a, z in zip(salt, saz)]
            stars = [rust.Star(float(v), bool(f)) for v, f in zip(vmag, found)]
            gen = rust.MagnitudeMapGenerator(coords, stars)
            grid = [rust.AltAzCoord(a, z) for a, z in zip(galt, gaz)]
            av = gen.query_many(grid, radius)
            return np.array([x.value if x is not None else np.nan for x in av])

        t_rs, res_rs = timeit(run_rust)
        both = np.isfinite(res_rs) & np.isfinite(res_np)
        agree = np.mean(np.abs(res_rs[both] - res_np[both]) < 1e-6)
        nan_agree = np.mean(np.isnan(res_rs) == np.isnan(res_np))
        line += f" | rust {t_rs * 1e3:8.1f} ms | agree {agree * 100:5.1f}% (nan {nan_agree * 100:5.1f}%)"
    print(line, flush=True)


def bench_query(rust: Any, n_points: int, rng: np.random.Generator) -> None:
    alt = np.arcsin(rng.uniform(0, 1, n_points))
    az = rng.uniform(0, 2 * np.pi, n_points)
    cloudy = rng.random(n_points) < 0.4
    centres = [(np.pi / 2, 0.0, np.pi / 2), (np.pi / 2, 0.0, np.deg2rad(10)), (0.8, 1.0, np.deg2rad(20)), (0.3, 4.0, np.deg2rad(20))]
    pts = rng.uniform(0, 1, (1000, 2)) * [np.pi / 2, 2 * np.pi]

    def run_np() -> list[float]:
        tree = cKDTree(to_xyz(alt, az))
        out = []
        for a, z, r in centres:
            idx = tree.query_ball_point(to_xyz(np.array([a]), np.array([z]))[0], chord(r))
            out.append(float(np.mean(cloudy[idx])))
        tree.query(to_xyz(pts[:, 0], pts[:, 1]))
        return out

    t_np, res_np = timeit(run_np, repeat=1 if n_points > 2_000_000 else 3)
    line = f"query points={n_points:>9d}  numpy {t_np * 1e3:8.1f} ms"
    if rust is not None and n_points <= 2_000_000:

        def run_rs() -> list[float]:
            q = rust.SkyPixelQuery([rust.AltAzCoord(a, z) for a, z in zip(alt, az)], [bool(c) for c in cloudy])
            out = [q.query_radius(rust.AltAzCoord(a, z), r) for a, z, r in centres]
            for a, z in pts:
                q.query_nearest_coordinate(rust.AltAzCoord(a, z))
            return out

        t_rs, res_rs = timeit(run_rs, repeat=1)
        line += f" | rust {t_rs * 1e3:8.1f} ms | max diff {np.max(np.abs(np.subtract(res_np, res_rs))):.2e}"
    print(line, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=None, help="RAYON_NUM_THREADS for the Rust code")
    p.add_argument("--no-rust", action="store_true")
    args = p.parse_args()
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    rust = None
    if not args.no_rust:
        import cloudmap_rs as rust  # type: ignore[import-not-found,no-redef]

    rng = np.random.default_rng(42)
    for grid_n, n_stars in [(100_000, 2_500), (100_000, 10_000), (400_000, 10_000), (400_000, 40_000)]:
        bench_night(rust, grid_n, n_stars, rng)
    for n in [25_000, 500_000, 1_500_000, 6_400_000, 12_300_000]:
        bench_query(rust, n, rng)


if __name__ == "__main__":
    main()
