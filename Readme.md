# pyobs allsky cloudcover 
[![pytest](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/pytest.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/pytest.yml)
[![ruff](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/ruff.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/ruff.yml)
[![mypy](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/mypy.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/mypy.yml)

This module analyzes Allsky images for cloud cover and writes the (zenith) cloud fraction and cloud change into an Influx DB.
It also includes a web service to query celestial positions for the limiting magnitude, which is calculated in an intermediate step.

Best paired with: [pyobs-allsky](https://gitlab.gwdg.de/iag/k.schimpf/pyobs-allsky)

## Installation

Pure Python (3.11+), installed with [uv](https://docs.astral.sh/uv/):

    uv sync
    uv run pytest

## Documentation

How it works, the web service's query routes, and the full configuration schema: see
[`docs/source/index.rst`](docs/source/index.rst) (built with Sphinx —
`cd docs && uv run --with sphinx --with sphinx-rtd-theme make html`).

## Configuration

See [example.yaml](https://github.com/pyobs/pyobs-allsky-cloudcover/blob/main/example.yaml)
