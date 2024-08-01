# pyobs allsky cloudcover 
[![Build](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/build.yml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/build.yml)
[![Tests](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/tests.yaml/badge.svg)](https://github.com/pyobs/pyobs-allsky-cloudcover/actions/workflows/tests.yaml)

This module analyzes Allsky images for cloud cover and writes the (zenith) cloud fraction and cloud change into an Influx DB.
It also includes a web service to query celestial positions for the limiting magnitude, which is calculated in an intermediate step.

Best paired with: [pyobs-allsky](https://gitlab.gwdg.de/iag/k.schimpf/pyobs-allsky)
## How does it work?
The module listens for a new image from the configured camera. If a new image has been captured, an image analysis pipeline is run on the image, depending on the solar altitude when the image was captured. Currently there is only a "night" pipeline, which works best after astronomical twilight (solar altitude below -18°).
This pipeline estimates the limiting magnitude of each pixel based on the visibility of the stars. Pixels are considered cloudy if their limiting magnitude falls below a defined threshold.

## Web Service

Route: `/query?ra={Right ascension in degree}&dec={Declination in degree}`

### Result
| Field    | Type  | Description                                     |
|----------|-------|-------------------------------------------------|
| obs_time | float | Observation Unix time of the last analyzed image |
| value    | float | Limiting magnitude in mag                       |


## Configuration

See [example.yaml](https://github.com/pyobs/pyobs-allsky-cloudcover/blob/main/example.yaml)