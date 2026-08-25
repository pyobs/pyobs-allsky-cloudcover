# pyobs allsky cloudcover 
[![Build](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/build.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/build.yml)
[![Tests](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/tests.yaml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/tests.yaml)

This module analyzes Allsky images for cloud cover and writes the (zenith) cloud fraction and cloud change into an Influx DB.
It also includes a web service to query celestial positions for the limiting magnitude, which is calculated in an intermediate step.

Best paired with: [pyobs-allsky](https://gitlab.gwdg.de/iag/k.schimpf/pyobs-allsky)

## Documentation

How it works, the web service's query routes, and the full configuration schema: see
[`docs/source/index.rst`](docs/source/index.rst) (built with Sphinx —
`cd docs && uv run --with sphinx --with sphinx-rtd-theme make html`).

## Configuration

See [example.yaml](https://github.com/pyobs/pyobs-allsky-cloudcover/blob/main/example.yaml)
