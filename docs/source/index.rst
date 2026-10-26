pyobs-allsky-cloudcover
########################

A `pyobs <https://www.pyobs.org>`_ module that estimates cloud cover from all-sky camera images. It works with
any camera, from a Raspberry Pi HQ camera to a cooled ZWO, mono or colour, at any resolution and bit depth: it
picks images up from the directory your capture software writes into (or from a pyobs camera module), writes
cloud fractions to InfluxDB and answers queries over HTTP.

.. toctree::
   :maxdepth: 1

   calibration
   simulator
   accuracy
   migration


How it works
************

Every new image goes through the **camera profile** (``camera`` section): it is read (FITS, PNG/TIFF/JPEG,
DNG), dark-subtracted and cleaned of hot pixels, turned into the canonical orientation, and given its Bayer
pattern, saturation level and an exact UTC timestamp. The **pipeline** for the sun's altitude at that time then
analyses it on a fixed grid of sky positions:

Night (sun below -18 deg by default)
    The catalogue stars above the horizon are projected into the image with the **world model** (the lens),
    and each is tested for a star at its position (aperture photometry against the local background). Around
    every sky position, the cut between the magnitudes of the seen and unseen stars within 7 deg is the
    **limiting magnitude**; below a camera-specific threshold the position is cloudy. Stars behind the sky
    mask aren't used; the area around the moon is masked.

Day (sun above -6 deg in the examples)
    Clear sky is blue and clouds are white or grey: the colour ratio blue/red + blue/green of the raw colours
    is thresholded; the area around the sun is masked. Needs a colour camera.

Twilight (in between)
    Neither method is reliable (a brighter sky looks like cloud to the night method, the sky loses its colour
    for the day method), so there is no result; ``/status`` says why.

Images are analysed one at a time off the event loop; if the analysis falls behind, only the newest image is
analysed. Results are queued for InfluxDB and retried if the database is down.


Installation and running
************************

::

    uv venv /opt/pyobs-cloudcover && uv pip install --python /opt/pyobs-cloudcover pyobs-allsky-cloudcover
    /opt/pyobs-cloudcover/bin/pyobs cloudcover.yaml

``deploy/`` has a systemd unit and a Dockerfile. To set up a camera, see :doc:`calibration`.


Configuration
*************

``example.yaml`` is a commented, generic config; ``examples/`` has complete configs for the camera presets
of the simulator. In outline::

    class: pyobs_cloudcover.application.Application
    watch:            # directory the capture software writes into (and/or image_sender: <camera module>)
      path: /var/lib/allsky/images
      pattern: "*.fits"
    camera:           # Bayer pattern, orientation, timestamps, dark, hot pixels, mask, world model
      ...
    server: {url: 0.0.0.0, port: 8080}
    measurement_log:  # InfluxDB (or memory/none) and which fields to write
      logger: {type: influx, url: ..., bucket: ..., org: ..., token: ...}
      measurements:
        cloud_total: [{type: total, name: cloudcover}]
        cloud_zenith: [{type: zenith, name: cloudcover, altitude: 80}]
    pipelines:
      night: {alt_interval: {end: -18}, options: {cloud_map: {threshold: 4.6}}}
      day: {alt_interval: {start: -6}, options: {cloud_map: {threshold: 1.85}}}
    location: {latitude: ..., longitude: ..., elevation: ...}

The config is validated at startup. Measurement field types: ``total`` (whole sky), ``zenith`` (above
``altitude``), ``sun``/``moon`` (within ``radius`` of it) and ``change`` (fraction of positions that changed
since the previous image). Fractions are 0-1.

Camera
======

``bayer``
    ``auto`` (``BAYERPAT``/``COLORTYP`` header keywords, honouring ``ROWORDER`` and ``X/YBAYROFF``), ``none``
    (mono) or the pattern of the stored array's first row (RGGB, BGGR, GRBG, GBRG).
``orientation``
    ``flip_x``, ``flip_y``, ``rotate`` (0/90/180/270): raw image to canonical image. Masks and the world model
    refer to the canonical image.
``timestamp``
    ``source`` (``auto``, ``header``, ``filename``, ``mtime``), ``keywords``, ``filename_format`` (strptime),
    ``timezone`` for timestamps without one, ``offset`` in seconds.
``saturation`` / ``bit_depth``
    Saturation level in ADU; guessed from the data if not given (left-shifted 12/14-bit data is recognised).
``dark``, ``hot_pixels``, ``mask``
    Files from ``pyobs-cloudcover darks|hot-pixels|mask``.
``world_model``
    ``FisheyeModel`` (equidistant, equisolid, stereographic, orthographic or polynomial lens; from
    ``pyobs-cloudcover calibrate``), ``SimpleModel`` or ``WCSModel`` (a FITS file with a WCS).

Night pipeline options
======================

``bin_size`` (2; even for colour cameras), ``catalog`` (``filepath`` empty for the bundled Bright Star
Catalogue, ``filter.alt``, ``filter.v_mag``), ``detector`` (``type: aperture`` with ``snr``, ``search_radius``,
``aperture_radius``, ``annulus``, ``max_background``; or ``type: sigma``, the 1.x detector), ``altaz_grid``
(``point_number``, ``limiting_altitude``), ``lim_mag_map`` (``radius``, ``estimator: robust|border``,
``integration_length``), ``cloud_map.threshold`` (limiting magnitude), ``moon_apparent_size``.

Day pipeline options
====================

``bin_size`` (2), ``cloud_map.threshold`` (colour ratio), ``sun_apparent_size``.


Web service
***********

All positions are degrees, either ``alt``/``az`` or ``ra``/``dec``. Every query answer has ``obs_time`` (Unix
time of the image the answer is based on), ``pipeline`` (``night``/``day``) and, if the latest image gave no
result, ``reason``.

``/query/point?alt=..&az=..``
    ``value``: whether the analysed position closest to the given one is cloudy (null if unknown).
``/query/area?alt=..&az=..&radius=..``
    ``fraction``: cloudy fraction (0-1) within ``radius``; ``value``: the same in percent (deprecated).
``/query/limiting_magnitude?alt=..&az=..``
    ``value``: limiting magnitude (V) at the closest analysed position (night only).
``/query/limiting_magnitude/area?alt=..&az=..&radius=..``
    ``mean``, ``min``, ``max`` of the limiting magnitude within ``radius``.
``/status``
    Frames received/analysed/skipped/without result/failed, last file, ``result_age`` (seconds since the
    observation the API serves), pipeline, ``reason``, last error.
``/cloud_map.png``, ``/limiting_magnitude.png``, ``/cloud_map.html``
    Sky maps (north up, east left) for dashboards.


Command-line tools
******************

``pyobs-cloudcover``
    ``calibrate``, ``mask``, ``darks``, ``hot-pixels``, ``replay``, ``catalog``: see :doc:`calibration`.
``pyobs-cloudcover-sim``
    Simulated all-sky images with ground truth: see :doc:`simulator`.
