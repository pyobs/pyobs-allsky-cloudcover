pyobs-allsky-cloudcover
########################

This is a `pyobs <https://www.pyobs.org>`_ module that analyzes allsky images for cloud cover and
writes the (zenith) cloud fraction and cloud change into InfluxDB. It also runs a small web
service to query cloudiness at a given celestial position — see `Web service`_ below. Best paired
with `pyobs-allsky <https://gitlab.gwdg.de/iag/k.schimpf/pyobs-allsky>`_.

Unlike the rest of the driver-module fleet, this repo is Python + Rust (a
`maturin <https://www.maturin.rs/>`_/PyO3 extension, see ``src/``) built with Poetry, not ``uv`` —
one page still covers it, but there's no autoclass reference here: building the Rust extension
just to import it for docs isn't worth the cost for a repo this size. See `Available classes`_
below for a plain description instead.


How it works
************

The module listens for a new image from the configured camera (``image_sender``). When a new
image arrives, an image-analysis pipeline runs on it, chosen by the sun's altitude at capture
time (``pipelines.day``/``pipelines.night`` in the config, each with an ``alt_interval``).
Currently only the **night** pipeline is implemented — it works best after astronomical twilight
(solar altitude below -18°): it estimates the limiting magnitude at each pixel from the visibility
of known stars (matched against a bundled catalog), and a pixel counts as cloudy once its limiting
magnitude falls below a threshold.


Example configuration
**********************

The full config schema is large (per-pipeline world model, catalog, star-matching, and cloud-map
options) — see `example.yaml
<https://github.com/pyobs/pyobs-allsky-cloudcover/blob/main/example.yaml>`_ in this repo for a
complete, commented example rather than a duplicate copy here that can drift out of sync. In
outline::

    class: pyobs_cloudcover.application.Application

    image_sender: "allskycam"     # camera module to listen to

    server:                        # the web service, see below
      url: "localhost"
      port: 8080

    measurement_log:               # what gets written to InfluxDB, and how
      logger:
        type: "influx"
        url: ""
        bucket: ""
        org: ""
        token: ""
      measurements:
        cloud_zenith: [...]
        cloud_total: [...]
        coverage: [...]

    pipelines:                     # per-solar-altitude image analysis, see "How it works"
      day: {alt_interval: {...}, options: {...}}
      night: {alt_interval: {...}, options: {...}}


Web service
***********

Point query
=============
Returns the cloudiness value of the analyzed sky position closest to the requested position.

**Route**: ``/query/point?ra=<degrees>&dec=<degrees>`` or ``/query/point?alt=<degrees>&az=<degrees>``

**Example**: ``/query/point?alt=90.0&az=0.0``

**Result**:

- ``obs_time`` (float) — observation Unix time of the last analyzed image.
- ``value`` (bool) — whether it's cloudy at the requested point.

Area query
============
Returns the cloud fraction within the requested great circle.

**Route**: ``/query/area?ra=<degrees>&dec=<degrees>&radius=<degrees>`` or
``/query/area?alt=<degrees>&az=<degrees>&radius=<degrees>``

**Example**: ``/query/area?alt=90.0&az=0.0&radius=10.0``

**Result**:

- ``obs_time`` (float) — observation Unix time of the last analyzed image.
- ``value`` (float) — cloud fraction in the requested area, in percent.


Available classes
******************

``pyobs_cloudcover.application.Application``
    The module class itself (see above).

``pyobs_cloudcover.world_model.SimpleModel`` / ``pyobs_cloudcover.world_model.WCSModel``
    The astrometric solution used to map pixels to sky positions — either a fitted analytic model
    (``a0``, ``F``, ``R``, ``c_x``, ``c_y`` — see `Berdina et al. 2019
    <https://ui.adsabs.harvard.edu/abs/2019A%26A...626A.105B>`_) or a WCS read from a FITS file.
    Selected via ``pipelines.<day|night>.options.world_model.class`` in the config.
