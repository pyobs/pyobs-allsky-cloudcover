# pyobs allsky cloudcover 
[![pytest](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/pytest.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/pytest.yml)
[![ruff](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/ruff.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/ruff.yml)
[![mypy](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/mypy.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/mypy.yml)

A [pyobs](https://www.pyobs.org) module that estimates cloud cover from all-sky camera images, for any camera
(Raspberry Pi HQ to cooled ZWO, mono or colour, any resolution, bit depth, Bayer pattern, orientation and lens).
It picks images up from the directory your capture software writes into (or from a pyobs camera module), writes
cloud fractions to InfluxDB, and serves cloud fractions, limiting magnitudes, status and sky-map images over HTTP.

- Night: stars from a catalogue are looked for in the image; the limiting magnitude around each sky position
  tells clear from cloudy.
- Day: the colour of the sky (blue = clear, white/grey = cloud).

## Installation

Pure Python (3.11+), with [uv](https://docs.astral.sh/uv/):

    uv sync
    uv run pytest

Run it with `pyobs config.yaml`; `deploy/` has a systemd unit and a Dockerfile.

## Setting up a camera

Zenith-centred fisheyes use a `FisheyeModel`; wide-angle or tilted cameras without a lens circle (e.g. a Raspberry
Pi Camera Module 3 Wide) use a `PointedModel` from `pyobs-cloudcover calibrate --pointed`. Cameras that get
bumped can be followed per frame with `pipelines.night.options.align` (see
[examples/rpi-cm3-wide-indi-allsky.yaml](examples/rpi-cm3-wide-indi-allsky.yaml), which also shows how to feed
it from indi-allsky's raw exports).

`pyobs-cloudcover calibrate | mask | darks | hot-pixels | replay | catalog` fit the lens, build the sky mask and
hot-pixel map and help choose thresholds from your own images: see
[docs/source/calibration.rst](docs/source/calibration.rst). Start from [example.yaml](example.yaml) or one of
the [examples](examples/).

## Simulator

`pyobs-cloudcover-sim` renders realistic all-sky FITS images with ground truth (stars, moon, sun, clouds,
noise, several camera presets); all tests run on it. See [docs/source/simulator.rst](docs/source/simulator.rst)
and the accuracy it gives: [docs/source/accuracy.rst](docs/source/accuracy.rst).

## Documentation

[docs/source/index.rst](docs/source/index.rst) (Sphinx: `cd docs && uv run --with sphinx --with sphinx-rtd-theme make html`).
Upgrading from 1.x: [docs/source/migration.rst](docs/source/migration.rst).
