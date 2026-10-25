Setting up a new camera
#######################

Everything camera specific lives in the ``camera`` section of the module config; the ``pyobs-cloudcover``
command fills it in from images. ``tests/integration/test_new_camera_setup.py`` runs exactly these steps on a
simulated camera.

1. What you know already
************************

Start a config with the site and how your capture software writes images (see ``examples/`` for complete
configs)::

    location: {latitude: -32.38, longitude: 20.81, elevation: 1760}
    watch: {path: /var/lib/allsky/images, pattern: "image-*.fits"}
    camera:
      bayer: auto          # or the pattern of the stored array (RGGB, ...) if the header has none; none = mono
      bit_depth: 12        # if the data aren't left-aligned in 16 bit (Raspberry Pi), else leave out
      orientation: {flip_x: false, flip_y: false, rotate: 0}   # any orientation works; see below
      timestamp: {source: auto, keywords: [DATE-OBS], timezone: UTC}
    pipelines:
      night: {alt_interval: {end: -18}, options: {}}
      day: {alt_interval: {start: 18}, options: {}}

Prefer raw FITS. 8-bit, stretched or JPEG images work, but lose the faint stars the night pipeline needs.

The orientation only defines the "canonical" image all other settings refer to; the world model absorbs
mirroring and rotation, so the default is fine unless you want the canonical image to look a certain way.

2. World model
**************

Take two or more clear, moonless night images a few hours apart and run::

    pyobs-cloudcover calibrate config.yaml night1.fits night2.fits

It finds the lens circle, detects stars and fits the lens projection, mirror parity, rotation, focal length and
centre (and a polynomial lens if that fits clearly better), printing a ``world_model`` block for the ``camera``
section and how far down the fit is constrained. It refuses (exit code 2) unless clearly more stars match than
at wrong rotations. If trees or buildings hide much of the horizon, add a few daytime images for the lens
circle: ``--circle-from day/*.fits``.

3. Sky mask
***********

From daytime images (several times of day, clear or partly cloudy)::

    pyobs-cloudcover mask config.yaml day/*.fits -o mask.npy

and set ``camera.mask: mask.npy``. Stars behind the mask are not used, so obstructions don't read as cloud.
``--min-alt`` additionally cuts everything below an altitude; ``--altitude-only`` does only that.

4. Darks and hot pixels
***********************

Without a shutter, build the hot-pixel map from ordinary night images (at least three, well apart in time)::

    pyobs-cloudcover hot-pixels config.yaml night/*.fits -o hot.npy

If you can cover the lens, a master dark (same exposure and temperature as the night images) is better::

    pyobs-cloudcover darks darks/*.fits -o dark.fits --hot-pixels hot.npy

Set ``camera.dark`` and ``camera.hot_pixels``. Hot pixels then no longer pass for stars.

5. Thresholds
*************

Let the camera run for a night or two with changing cloud (or point ``replay`` at an archive), then::

    pyobs-cloudcover replay config.yaml archive/ --csv night.csv --plot night.png

It analyses every image, writes the cloud fractions to a CSV and suggests a threshold per pipeline: the value
that best separates the two populations (clear and cloudy) of limiting magnitudes or colour ratios. On
simulated data this is within 0.02 in accuracy of the best possible threshold, but it needs both clear and
cloudy images. Put it into ``pipelines.night.options.cloud_map.threshold`` (and ``day``). The plot (needs
``matplotlib``) shows the histograms with the suggested cuts.

6. Deeper catalogue (optional)
******************************

The bundled Bright Star Catalogue stops near V = 6.5. Sensitive cameras (limiting magnitude > 6) benefit from
more stars::

    pyobs-cloudcover catalog --max-mag 8      # needs astroquery: pip install pyobs-allsky-cloudcover[catalog]

and set ``catalog.filepath`` to the file it writes (~1 MB, ~40 000 stars) and ``catalog.filter.v_mag: 8``.
