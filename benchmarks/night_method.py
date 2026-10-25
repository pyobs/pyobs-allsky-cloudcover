"""Compare star detectors and limiting-magnitude estimators for the night pipeline on simulated images.

Same protocol as benchmarks/star_weight.py: cameras x cloud covers x optical depths, thresholds trained on
other seeds, scored against ground truth on the sky grid (alt >= 30 deg).

Run:  uv run python benchmarks/night_method.py [--scale 0.5] [--seeds 2]
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from star_weight import COVERS, DEPTHS, PRESETS, RADIUS, TIMES, score

from pyobs_cloudcover.data import bundled_catalog_path
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.border_magnitude import (
    border_magnitude,
    robust_border_magnitude,
)
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_factory import (
    StarReverseMatcherFactory,
)
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import (
    StarReverseMatcherOptions,
)
from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

legacy_detector = StarReverseMatcherFactory(
    StarReverseMatcherOptions(sigma_threshold=3.0, window_size=6, median_limit=1e12)
)()
DETECTORS: dict[str, Callable[..., list[bool]]] = {
    "sigma (legacy)": legacy_detector,
    "aperture snr3": ApertureMatcher(snr=3.0),
    "aperture snr4": ApertureMatcher(snr=4.0),
    "aperture snr5": ApertureMatcher(snr=5.0),
}
ESTIMATORS = {"border": border_magnitude, "robust": robust_border_magnitude}


def analyse(preset: str, scale: float, cover: float, depth: float, seed: int) -> dict[str, np.ndarray]:
    sim = Simulator(
        get_preset(preset, scale), clouds=[CloudLayer(cover=cover, optical_depth=depth, seed=seed)], seed=seed
    )
    time = TIMES[seed % len(TIMES)]
    frame = sim.render(time)
    image = sim.profile.orientation.to_canonical(frame.raw.astype(np.float64))
    binned = ImageBinner(2)(image)
    model = sim.world_model.scaled(0.5)
    catalog = CatalogConstructor(
        AltAzCatalogLoader.from_csv(bundled_catalog_path()), model, sim._observer, 30.0, 7.0, 0.0
    )(time, *binned.shape)
    grid = SphericalAltAzGenerator(100000, 30)()
    out = {
        "truth": frame.truth.cloudy_at(np.rad2deg(grid.alt), np.rad2deg(grid.az)),
        "zenith": grid.alt >= np.deg2rad(80),
    }
    for dname, detector in DETECTORS.items():
        found = np.array(detector(binned, catalog), dtype=bool)
        for ename, estimator in ESTIMATORS.items():
            out[f"{dname} + {ename}"] = estimator(
                grid, np.deg2rad(catalog.alt), np.deg2rad(catalog.az), catalog.v_mag, found, RADIUS
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args()

    thresholds = np.arange(2.5, 7.01, 0.05)
    methods = [f"{d} + {e}" for d in DETECTORS for e in ESTIMATORS]
    results: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    chosen: dict[tuple[str, str], float] = {}
    for preset in PRESETS:
        runs = {
            (cover, depth, seed): analyse(preset, args.scale, cover, depth, seed)
            for cover, depth, seed in itertools.product(COVERS, DEPTHS, range(2 * args.seeds))
        }
        train = [k for k in runs if k[2] < args.seeds]
        test = [k for k in runs if k[2] >= args.seeds]
        for name in methods:
            agreement = [
                np.mean([score(runs[k][name], runs[k]["truth"], runs[k]["zenith"], t)[0] for k in train])
                for t in thresholds
            ]
            best = float(thresholds[int(np.argmax(agreement))])
            chosen[(preset, name)] = best
            for k in test:
                results[(preset, name)].append(score(runs[k][name], runs[k]["truth"], runs[k]["zenith"], best))
        print(f"done {preset}", flush=True)

    print()
    print(f"{'camera':14s} {'method':28s} {'thresh':>6s} {'agreement':>9s} {'|total err|':>11s} {'|zenith err|':>12s}")
    overall: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (preset, name), rows in results.items():
        a, te, ze = (float(np.nanmean([r[i] for r in rows])) for i in range(3))
        overall[name].append((a, te, ze))
        print(f"{preset:14s} {name:28s} {chosen[(preset, name)]:6.2f} {a:9.3f} {te:11.3f} {ze:12.3f}")
    print()
    for name, rows in overall.items():
        a, te, ze = (float(np.mean([r[i] for r in rows])) for i in range(3))
        print(f"{'all':14s} {name:28s} {'':6s} {a:9.3f} {te:11.3f} {ze:12.3f}")


if __name__ == "__main__":
    main()
