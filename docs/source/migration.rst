Migrating from 1.x
##################

Configs written for 1.x keep working: a ``pipelines.night.options.preprocessor`` section (or
``pipelines.day.options.mask_filepath``) marks a pipeline as 1.x style, and it is then run exactly as
before (same detector, same limiting-magnitude estimator, same coordinate conventions), so tuned
thresholds keep their meaning. The only differences are bug fixes (see the changelog). The bundled star
catalogue moved into the package; ``catalog.filepath: data/catalog_v50.csv`` still finds it.

The new style is recommended for new installations and needed for anything but the old camera setup.
To convert a 1.x config:

1. **Image source.** ``image_sender`` still works. To read images from a directory instead (no
   pyobs camera module needed), add a ``watch`` section (``path``, ``pattern``).

2. **Camera section.** Move the ``world_model`` out of the pipelines into a top-level ``camera`` section.
   Its coordinates change, though:

   * 1.x night models were fitted on the image *flipped vertically* and *binned 2x2*. New-style models use
     the full-resolution image in the camera's canonical orientation (``camera.orientation``; by default
     the image as stored). The flip mirrors the projection, which ``SimpleModel`` can't express, so
     convert to a ``FisheyeModel``. For a ``SimpleModel`` with ``F: 2`` fitted with ``bin_size: 2`` on
     an image of full-resolution height ``H``, the equivalent model is::

        class: pyobs_cloudcover.world_model.FisheyeModel
        projection: equisolid
        parity: -1
        a0: a0_old
        f: R_old
        c_x: 2 * c_x_old + 0.5
        c_y: H - 1 - (2 * c_y_old + 0.5)

     Other values of ``F`` have no exact equivalent; refit with ``pyobs-cloudcover calibrate``.
   * 1.x day models were in full-resolution raw pixels already; they carry over unchanged.

3. **Mask.** ``preprocessor.mask_filepath`` (night, flipped frame) and ``mask_filepath`` (day) become
   ``camera.mask`` (canonical frame, nonzero = sky). A mask is no longer multiplied into the image:
   stars outside it are simply not used, so obstructions no longer read as clouds.

4. **Night pipeline.** Replace ``preprocessor`` by ``bin_size``. The detector (``detector:``) and the
   estimator (``lim_mag_map.estimator``) default to ``aperture`` and ``robust``, which are considerably more
   accurate on simulated data; the limiting-magnitude threshold then needs re-tuning (typically 4-5.3 mag
   instead of 5.5), e.g. with ``pyobs-cloudcover replay``. To keep the 1.x method in a new-style config,
   set ``detector: {type: sigma, sigma_threshold: 3, window_size: 6}`` and
   ``lim_mag_map: {estimator: border}``. ``reverse_matcher.median_limit`` (ADU) becomes
   ``detector.max_background`` (fraction of the saturation level).

5. **Bayer pattern.** 1.x hardcoded the day pipeline's debayering (RGGB at the array origin). It now comes
   from the ``BAYERPAT``/``COLORTYP`` header keywords (honouring ``ROWORDER``) or ``camera.bayer``.

6. **Units.** ``/query/area`` gained ``fraction`` (0-1); ``value`` stays in percent but is deprecated.
