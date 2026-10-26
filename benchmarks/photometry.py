"""Photometric mode vs. the limiting-magnitude method on simulated images.

For each camera, moon phase and cloud scenario: agreement of the cloud map from the limiting magnitude (the
camera's tuned threshold) and from the measured extinction (>= 0.5 mag, the same for every camera), the mean
absolute error of the extinction map against the true cloud extinction, and the zenith sky brightness against
the simulator's.

    uv run python benchmarks/photometry.py [--scale 1.0]
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict

import numpy as np
import yaml
from accuracy import ROOT

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.evaluation import NIGHTS, Scenario, evaluate, run_scenario
from pyobs_cloudcover.simulation.truth import GroundTruth

PRESETS = ("zwo-asi224mc", "rpi-hq", "zwo-asi178mm")
CLOUDS = {"clear": (0.0, 3.0), "half": (0.4, 3.0), "overcast": (1.0, 3.0), "cirrus": (0.6, 0.8)}


def extinction_scores(info: CloudCoverageInfo, truth: GroundTruth) -> tuple[float, float]:
    q = info.extinction_query
    assert q is not None
    alt, az = np.rad2deg(q.grid.alt), np.rad2deg(q.grid.az)
    ext = q.get_pixels()
    ok = np.isfinite(ext) & (alt >= 30)
    true = np.clip(truth.extinction_at(alt[ok], az[ok]), 0, 3)
    cloudy = ext[ok] >= 0.5
    return float(np.mean(cloudy == (true >= truth.cloud_threshold))), float(np.mean(np.abs(ext[ok] - true)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for preset in PRESETS:
            config = yaml.safe_load((ROOT / "examples" / f"{preset}.yaml").read_text())
            # as in operation: the zero point is known from clear frames (here: one clear, dark frame)
            clear, _ = run_scenario(Scenario(preset, "dark", 0.0, seed=99), config, tmp, args.scale)
            assert clear is not None and clear.zero_point is not None
            options = config["pipelines"]["night"]["options"]
            options["photometry"] = {**options.get("photometry", {}), "zero_point": clear.zero_point}
            print(f"{preset}: zero point {clear.zero_point:.2f}", flush=True)
            for sky in NIGHTS:
                for name, (cover, depth) in CLOUDS.items():
                    info, truth = run_scenario(Scenario(preset, sky, cover, depth), config, tmp, args.scale)
                    assert info is not None
                    # the limiting-magnitude cloud map with the camera's threshold (the default map is photometric)
                    m = evaluate(info, truth, threshold=options["cloud_map"]["threshold"])
                    assert m is not None
                    agree_ext, mae = extinction_scores(info, truth)
                    true_mag = truth.metadata["sky_zenith_mag"]
                    rows.append((preset, sky, name, m.agreement, agree_ext, mae, info.sky_brightness, true_mag))
                    print(
                        f"{preset:13s} {sky:12s} {name:8s} lim-mag {m.agreement:.2f}  extinction {agree_ext:.2f}  "
                        f"|ext err| {mae:.2f} mag  sky {info.sky_brightness or float('nan'):.2f} "
                        f"(true {true_mag:.2f})",
                        flush=True,
                    )
    print()
    by: dict[str, list] = defaultdict(list)  # type: ignore[type-arg]
    for r in rows:
        by[r[1]].append(r)
        by[r[2]].append(r)
    for key, rs in by.items():
        sky_err = [abs(r[6] - r[7]) for r in rs if r[6] is not None]
        lim, ext, err = (np.mean([r[i] for r in rs]) for i in (3, 4, 5))
        sky = np.mean(sky_err) if sky_err else np.nan
        print(f"{key:12s} agreement lim-mag {lim:.2f}  extinction {ext:.2f}  |ext err| {err:.2f}  |sky err| {sky:.2f}")


if __name__ == "__main__":
    main()
