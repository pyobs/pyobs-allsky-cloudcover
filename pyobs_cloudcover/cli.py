"""``pyobs-cloudcover``: set up a camera and check the analysis.

Most commands take the module config (YAML) for the site and the ``camera`` section (Bayer pattern,
orientation, timestamps); they don't need a world model yet, except ``mask`` and ``replay``.

    pyobs-cloudcover calibrate config.yaml night/*.fits      # fit the world model
    pyobs-cloudcover mask config.yaml day/*.fits -o mask.npy  # sky mask
    pyobs-cloudcover darks darks/*.fits -o dark.fits          # master dark (+ hot pixels)
    pyobs-cloudcover hot-pixels config.yaml night/*.fits -o hot.npy
    pyobs-cloudcover replay config.yaml images/ --csv out.csv --plot out.png
    pyobs-cloudcover catalog --max-mag 8                      # deeper star catalogue
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import yaml
from astroplan import Observer

log = logging.getLogger("pyobs_cloudcover.cli")


def load_config(path: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"{path} is not a module config")
    return config


def observer_from(config: dict[str, Any]) -> Observer:
    loc = config.get("location")
    if not loc:
        raise SystemExit("The config needs a 'location' (latitude, longitude, elevation)")
    return Observer(
        latitude=float(loc["latitude"]) * u.deg,
        longitude=float(loc["longitude"]) * u.deg,
        elevation=float(loc.get("elevation", 0.0)) * u.m,
    )


def expand(paths: Sequence[str], pattern: str = "*.fits") -> list[str]:
    files: list[str] = []
    for p in paths:
        path = Path(p)
        files.extend(sorted(str(f) for f in path.glob(pattern)) if path.is_dir() else [p])
    if not files:
        raise SystemExit("No images")
    return files


def cmd_calibrate(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.astrometry import calibrate
    from pyobs_cloudcover.camera import CameraProfile, read_image

    config = load_config(args.config)
    camera = dict(config.get("camera") or {})
    camera.pop("world_model", None)
    profile = CameraProfile(camera)
    frames = [profile.prepare(read_image(p)) for p in expand(args.images)]
    circle = [profile.prepare(read_image(p)) for p in expand(args.circle_from)] if args.circle_from else None
    result = calibrate(frames, observer_from(config), polynomial=args.polynomial, circle_frames=circle)
    print(
        f"# {result.model.projection} lens, rms {result.rms:.2f} px over {result.matched} stars, "
        f"{100 * result.match_fraction:.0f} % of bright stars matched "
        f"({100 * result.baseline_fraction:.0f} % at wrong rotations); constrained down to "
        f"{result.lowest_altitude:.0f} deg altitude"
    )
    print(yaml.safe_dump({"world_model": _plain(result.config())}, sort_keys=False))
    if result.lowest_altitude > 30:
        print(f"# set pipelines.night.options.catalog.filter.alt to >= {result.lowest_altitude:.0f}")
    if not result.trustworthy:
        print(
            "# NOT TRUSTWORTHY: too few stars matched. Use clear, moonless frames and check the camera section "
            "(orientation, Bayer pattern, timestamps).",
            file=sys.stderr,
        )
        return 0 if args.force else 2
    return 0


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _model(config: dict[str, Any]) -> Any:
    from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory

    model_config = (config.get("camera") or {}).get("world_model")
    if not model_config:
        raise SystemExit("The config's camera section needs a world_model (run 'calibrate' first)")
    return WorldModelFactory(observer_from(config))(model_config)


def cmd_mask(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.mask import sky_mask_from_altitude, sky_mask_from_day
    from pyobs_cloudcover.camera import CameraProfile, read_image

    config = load_config(args.config)
    camera = dict(config.get("camera") or {})
    camera.pop("mask", None)
    profile = CameraProfile(camera)
    frames = [profile.prepare(read_image(p)) for p in expand(args.images)]
    model = _model(config)
    if args.altitude_only:
        mask = sky_mask_from_altitude(frames[0].shape, model, args.min_alt)
    else:
        mask = sky_mask_from_day(frames, model, min_alt=args.min_alt)
    np.save(args.output, mask)
    print(f"Wrote {args.output}: {100 * mask.mean():.1f} % of the image is sky. Set camera.mask: {args.output}")
    return 0


def cmd_darks(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.darks import hot_pixels_from_dark, master_dark, write_dark
    from pyobs_cloudcover.camera import read_image

    dark, exposure = master_dark([read_image(p) for p in expand(args.images)])
    write_dark(args.output, dark, exposure)
    print(f"Wrote {args.output} (exposure {exposure} s). Set camera.dark: {args.output}")
    if args.hot_pixels:
        hot = hot_pixels_from_dark(dark, args.sigma)
        np.save(args.hot_pixels, hot)
        print(f"Wrote {args.hot_pixels}: {int(hot.sum())} hot pixels. Set camera.hot_pixels: {args.hot_pixels}")
    return 0


def cmd_hot_pixels(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.darks import hot_pixels_from_frames
    from pyobs_cloudcover.camera import read_image

    hot = hot_pixels_from_frames([read_image(p) for p in expand(args.images)], args.sigma)
    np.save(args.output, hot)
    print(f"Wrote {args.output}: {int(hot.sum())} hot pixels. Set camera.hot_pixels: {args.output}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.replay import plot, replay, suggest_thresholds, suggest_zero_point, write_csv

    config = load_config(args.config)
    pattern = args.pattern or (config.get("watch") or {}).get("pattern", "*.fits")
    files = expand(args.images, pattern)

    def progress(r: Any) -> None:
        if r.error:
            print(f"{Path(r.path).name}: {r.error}")
        elif r.info is None:
            print(f"{Path(r.path).name}: no pipeline for this sun altitude")
        else:
            total = r.info.cloud_cover_query.query_radius(np.pi / 2, 0.0, np.deg2rad(60))
            print(
                f"{Path(r.path).name}: {r.info.pipeline} cloud fraction {total if total is None else round(total, 2)}"
            )

    results = replay(config, files, observer_from(config), progress)
    if args.csv:
        write_csv(results, args.csv)
    suggestions = suggest_thresholds(results)
    for s in suggestions:
        extra = f", agreement {s.agreement:.2f}" if s.agreement is not None else ""
        print(f"suggested {s.pipeline} threshold: {s.threshold:.2f} ({s.method}, {s.frames} frames{extra})")
    zero_point = suggest_zero_point(results)
    if zero_point is not None:
        print(f"photometric zero point: {zero_point:.2f} (pipelines.night.options.photometry.zero_point)")
    if args.plot:
        try:
            plot(results, suggestions, args.plot)
        except ImportError:
            print("Plots need matplotlib (uv pip install matplotlib)")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    from pyobs_cloudcover.calibration.catalog import fetch_hipparcos

    path = fetch_hipparcos(args.max_mag, args.output)
    print(f"Wrote {path}. Use pipelines.night.options.catalog.filepath: {path} and filter.v_mag: {args.max_mag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyobs-cloudcover", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("calibrate", help="fit the world model from clear-night images")
    s.add_argument("config")
    s.add_argument("images", nargs="+", help="clear, moonless night images (or directories), ideally hours apart")
    s.add_argument(
        "--polynomial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force (or forbid) a polynomial lens; default: use it if it fits clearly better",
    )
    s.add_argument("--force", action="store_true", help="exit 0 even if the fit isn't trustworthy")
    s.add_argument(
        "--circle-from",
        nargs="+",
        default=None,
        metavar="IMAGE",
        help="find the lens circle in these (e.g. daytime) images; helps when the horizon is obstructed",
    )
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("mask", help="build the sky mask from daytime images")
    s.add_argument("config")
    s.add_argument("images", nargs="+")
    s.add_argument("-o", "--output", default="mask.npy")
    s.add_argument("--min-alt", type=float, default=0.0, help="mask everything below this altitude (deg)")
    s.add_argument("--altitude-only", action="store_true", help="only cut at --min-alt, ignore the images")
    s.set_defaults(func=cmd_mask)

    s = sub.add_parser("darks", help="master dark from dark frames (lens covered)")
    s.add_argument("images", nargs="+")
    s.add_argument("-o", "--output", default="dark.fits")
    s.add_argument("--hot-pixels", default="", help="also write a hot-pixel map (.npy)")
    s.add_argument("--sigma", type=float, default=8.0)
    s.set_defaults(func=cmd_darks)

    s = sub.add_parser("hot-pixels", help="hot-pixel map from night images (at least 3, well apart in time)")
    s.add_argument("config")
    s.add_argument("images", nargs="+")
    s.add_argument("-o", "--output", default="hot_pixels.npy")
    s.add_argument("--sigma", type=float, default=6.0)
    s.set_defaults(func=cmd_hot_pixels)

    s = sub.add_parser("replay", help="analyse stored images, write a CSV/plot, suggest thresholds")
    s.add_argument("config")
    s.add_argument("images", nargs="+", help="images or directories")
    s.add_argument("--pattern", default="", help="glob in directories (default: the config's watch pattern)")
    s.add_argument("--csv", default="")
    s.add_argument("--plot", default="", help="PNG with time series and histograms (needs matplotlib)")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("catalog", help="download a deeper star catalogue (Hipparcos) for sensitive cameras")
    s.add_argument("--max-mag", type=float, default=8.0)
    s.add_argument("-o", "--output", default=None)
    s.set_defaults(func=cmd_catalog)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
