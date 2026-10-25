"""``pyobs-cloudcover-sim``: write simulated all-sky FITS images (plus ground truth) to a directory."""

from __future__ import annotations

import argparse
import datetime
import logging
import time
from collections.abc import Sequence

from pyobs_cloudcover.simulation.clouds import CloudLayer
from pyobs_cloudcover.simulation.profile import PRESETS, get_preset
from pyobs_cloudcover.simulation.render import Simulator
from pyobs_cloudcover.simulation.scene import Horizon, Site, SkyConditions

log = logging.getLogger(__name__)


def parse_time(value: str) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(value)
    return t if t.tzinfo else t.replace(tzinfo=datetime.UTC)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyobs-cloudcover-sim", description=__doc__)
    p.add_argument("output", help="Directory to write images into")
    p.add_argument("--preset", default="zwo-asi224mc", choices=sorted(PRESETS), help="Camera preset")
    p.add_argument("--scale", type=float, default=1.0, help="Resample the camera (e.g. 0.25 for quick tests)")
    p.add_argument("--time", type=parse_time, default=None, help="UTC time of the first frame (ISO 8601)")
    p.add_argument("--count", type=int, default=1, help="Number of frames")
    p.add_argument("--cadence", type=float, default=60.0, help="Seconds between frames (simulated time)")
    p.add_argument("--realtime", action="store_true", help="Wait --cadence seconds between frames (fake camera)")
    p.add_argument("--latitude", type=float, default=Site.latitude)
    p.add_argument("--longitude", type=float, default=Site.longitude)
    p.add_argument("--elevation", type=float, default=Site.elevation)
    p.add_argument("--timezone", default=Site.timezone)
    p.add_argument("--cover", type=float, default=0.3, help="Cloud cover of the layer, 0..1")
    p.add_argument("--optical-depth", type=float, default=3.0, help="Cloud extinction in mag (0.3 = thin cirrus)")
    p.add_argument("--wind", type=float, nargs=2, default=(8.0, 3.0), metavar=("EAST", "NORTH"), help="m/s")
    p.add_argument("--sky-mag", type=float, default=SkyConditions.zenith_mag, help="Dark zenith mag/arcsec^2")
    p.add_argument("--horizon", action="store_true", help="Add trees and a building on the horizon")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-truth", action="store_true", help="Don't write the .truth.npz sidecar files")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    site = Site(
        latitude=args.latitude,
        longitude=args.longitude,
        elevation=args.elevation,
        timezone=args.timezone,
        name="custom",
    )
    clouds = [CloudLayer(cover=args.cover, optical_depth=args.optical_depth, wind=tuple(args.wind), seed=args.seed)]
    sim = Simulator(
        get_preset(args.preset, args.scale),
        site=site,
        sky=SkyConditions(zenith_mag=args.sky_mag),
        clouds=clouds if args.cover > 0 else [],
        horizon=Horizon.trees_and_building(args.seed) if args.horizon else None,
        seed=args.seed,
    )
    start = args.time or datetime.datetime.now(datetime.UTC)
    for i in range(args.count):
        t = start + datetime.timedelta(seconds=i * args.cadence)
        frame = sim.render(t)
        path = frame.write(args.output, truth=not args.no_truth)
        meta = frame.truth.metadata
        log.info(
            "%s  sun %+.1f  moon %+.1f  exposure %.3gs  cloud fraction (alt>30) %.2f",
            path,
            meta["sun_alt"],
            meta["moon_alt"],
            meta["exposure"],
            frame.truth.cloud_fraction(30),
        )
        if args.realtime and i + 1 < args.count:
            time.sleep(args.cadence)


if __name__ == "__main__":
    main()
