"""Which star weight should the limiting-magnitude map use? Evidence from simulated images.

For a matrix of cameras, cloud covers and optical depths this renders frames, runs the night pipeline's
star matching (unchanged), and builds the limiting-magnitude map with different distance weights for the
stars within the search radius. Each map is thresholded into cloudy/clear and compared with the simulator's
ground truth on the sky grid (alt >= 30 deg).

The threshold is chosen per camera and weight on *training* seeds (best per-point agreement) and evaluated on
different *test* seeds, so no kernel profits from a threshold tuned on its own test data.

Run:  uv run python benchmarks/star_weight.py [--scale 0.5] [--seeds 2]
"""

from __future__ import annotations

import argparse
import datetime
import itertools
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from pyobs_cloudcover.data import bundled_catalog_path
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.border_magnitude import (
    border_magnitude,
    legacy_weight,
)
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_factory import (
    StarReverseMatcherFactory,
)
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import (
    StarReverseMatcherOptions,
)
from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

RADIUS = np.deg2rad(7.0)

WEIGHTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "legacy (w = -d*pi/1260)": legacy_weight,
    "uniform": lambda d: np.ones_like(d),
    "exp(-d/7deg)": lambda d: np.exp(-d / RADIUS),
    "exp(-d/3.5deg)": lambda d: np.exp(-d / (RADIUS / 2)),
    "gauss(sigma=3.5deg)": lambda d: np.exp(-0.5 * (d / (RADIUS / 2)) ** 2),
}

PRESETS = ("zwo-asi224mc", "rpi-hq", "zwo-asi178mm")
COVERS = (0.1, 0.3, 0.5, 0.7)
DEPTHS = (3.0, 1.0)
# a few moonless nights at the default site
TIMES = [datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC) + datetime.timedelta(days=d) for d in (0, 2, 4)]


def analyse(preset: str, scale: float, cover: float, depth: float, seed: int) -> dict[str, np.ndarray]:
    sim = Simulator(
        get_preset(preset, scale), clouds=[CloudLayer(cover=cover, optical_depth=depth, seed=seed)], seed=seed
    )
    time = TIMES[seed % len(TIMES)]
    frame = sim.render(time)
    profile = sim.profile

    image = profile.orientation.to_canonical(frame.raw.astype(np.float64))
    binned = ImageBinner(2)(image)
    model = sim.world_model.scaled(0.5)

    constructor = CatalogConstructor(
        AltAzCatalogLoader.from_csv(bundled_catalog_path()), model, sim._observer, 30.0, 7.0, 0.0
    )
    catalog = constructor(time, *binned.shape)
    matcher = StarReverseMatcherFactory(
        StarReverseMatcherOptions(sigma_threshold=3.0, window_size=6, median_limit=1e12)
    )()
    found = np.array(matcher(binned, catalog), dtype=bool)

    grid = SphericalAltAzGenerator(100000, 30)()
    truth = frame.truth.cloudy_at(np.rad2deg(grid.alt), np.rad2deg(grid.az))
    maps = {
        name: border_magnitude(grid, np.deg2rad(catalog.alt), np.deg2rad(catalog.az), catalog.v_mag, found, RADIUS, w)
        for name, w in WEIGHTS.items()
    }
    return {"truth": truth, "zenith": grid.alt >= np.deg2rad(80), **maps}


def score(lim: np.ndarray, truth: np.ndarray, zenith: np.ndarray, threshold: float) -> tuple[float, float, float]:
    known = np.isfinite(lim)
    cloudy = lim < threshold
    agree = float(np.mean(cloudy[known] == truth[known]))
    total_err = abs(float(np.mean(cloudy[known])) - float(np.mean(truth[known])))
    z = known & zenith
    zen_err = abs(float(np.mean(cloudy[z])) - float(np.mean(truth[z]))) if np.any(z) else np.nan
    return agree, total_err, zen_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=2, help="seeds per scenario, for training and for testing each")
    args = p.parse_args()

    thresholds = np.arange(3.0, 7.01, 0.1)
    results: dict[tuple[str, str], list[tuple[str, float, float, float]]] = defaultdict(list)

    for preset in PRESETS:
        runs = {
            (cover, depth, seed): analyse(preset, args.scale, cover, depth, seed)
            for cover, depth, seed in itertools.product(COVERS, DEPTHS, range(2 * args.seeds))
        }
        train = [k for k in runs if k[2] < args.seeds]
        test = [k for k in runs if k[2] >= args.seeds]
        for name in WEIGHTS:
            agreement = [
                np.mean([score(runs[k][name], runs[k]["truth"], runs[k]["zenith"], t)[0] for k in train])
                for t in thresholds
            ]
            best = float(thresholds[int(np.argmax(agreement))])
            for k in test:
                a, te, ze = score(runs[k][name], runs[k]["truth"], runs[k]["zenith"], best)
                results[(preset, name)].append((f"{k[0]}/{k[1]}", a, te, ze))
            print(f"{preset:14s} {name:24s} threshold {best:.1f}", flush=True)

    print()
    print(f"{'camera':14s} {'weight':24s} {'agreement':>9s} {'|total err|':>11s} {'|zenith err|':>12s}")
    overall: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (preset, name), rows in results.items():
        a, te, ze = (np.nanmean([r[i] for r in rows]) for i in (1, 2, 3))
        overall[name].append((a, te, ze))
        print(f"{preset:14s} {name:24s} {a:9.3f} {te:11.3f} {ze:12.3f}")
    print()
    for name, rows in overall.items():
        a, te, ze = (np.mean([r[i] for r in rows]) for i in range(3))
        print(f"{'all':14s} {name:24s} {a:9.3f} {te:11.3f} {ze:12.3f}")


if __name__ == "__main__":
    main()
