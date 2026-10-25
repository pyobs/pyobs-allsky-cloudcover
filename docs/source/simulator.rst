All-sky image simulator
#######################

``pyobs_cloudcover.simulation`` renders all-sky FITS images that the module ingests unchanged, together
with the ground truth that went into them. All tests run on its output; there are no image fixtures.

Quick start::

    # one frame now, from the default camera, into ./sim
    uv run pyobs-cloudcover-sim ./sim

    # act as a fake camera: a frame every 60 s through a night, half-size images
    uv run pyobs-cloudcover-sim ./sim --preset rpi-hq --scale 0.5 --time 2026-03-20T18:00 \
        --count 600 --cadence 60 --realtime --cover 0.4 --horizon

From Python::

    from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

    sim = Simulator(get_preset("zwo-asi224mc"), clouds=[CloudLayer(cover=0.4, optical_depth=3)], seed=1)
    frame = sim.render(datetime.datetime(2026, 3, 20, 22, tzinfo=datetime.UTC))
    frame.write("sim/")              # image + <name>.truth.npz
    frame.truth.cloud_fraction(30)   # true cloudy fraction above 30 deg


What is modelled
****************

Camera
    Sensor size, bit depth and storage (e.g. 12-bit data left-shifted into 16-bit), Bayer pattern or mono,
    gain, read noise, bias, dark current, hot pixels (fixed per camera), full well and ADC saturation,
    optional gamma stretch (cheap cameras), PSF widening towards the horizon, vignetting, the raw sensor
    orientation (flips/rotation) and the lens as a ``FisheyeModel`` (equidistant, equisolid,
    stereographic, orthographic or polynomial).

Sky
    Krisciunas & Schaefer (1991) dark sky growing with airmass, scattered moonlight, light pollution and an
    optional light dome; an empirical table of zenith brightness against sun altitude for twilight and
    daylight, brighter towards the sun and horizon; a blue clear-sky colour in daylight that whitens towards
    the horizon and the sun. Atmospheric extinction (mag per airmass).

Stars, moon and sun
    The bundled Yale Bright Star Catalogue (the stars the night pipeline tests) plus synthetic fainter stars
    down to ``faint_star_limit`` (V = 8.5 by default) with a realistic number-magnitude slope; they confuse
    the detector the way real faint stars do. Flux from magnitude, zero point, extinction, clouds and
    vignetting. Moon and sun are discs of their apparent size and brightness.

Clouds
    Fractal layers at a height above the camera, so clouds shrink towards the horizon; cover, optical depth
    (thin cirrus ~0.3 mag to opaque), softness and drift with the wind between frames. At night they hide
    the sky glow above them and reflect ground light and moonlight; by day thin clouds are bright white and
    thick ones grey.

Horizon
    Obstructions (``Horizon``: altitude against azimuth) block stars and appear as dark silhouettes.

Exposure
    Auto-exposure like capture software (median sky at <= 15 % of saturation), so twilight and daylight
    frames don't saturate. Poisson, read and quantisation noise.

Headers and file names
    Each preset imitates some capture software (INDI, N.I.N.A., Raspberry Pi scripts, a minimal one with no
    timestamp at all) with different keywords, timestamp formats and file-name patterns, so header and
    timestamp parsing are genuinely exercised. Files are written atomically.


Ground truth
************

Next to every image, ``<name>.truth.npz`` (load with ``GroundTruth.load``) holds

* the cloud extinction (mag) on a 1 deg x 2 deg alt/az grid, hence the transmission and the cloudy map
  (extinction >= ``cloud_threshold``, 0.5 mag by default);
* every catalogue star above the horizon: position (alt/az and canonical pixels), flux, cloud
  extinction, expected SNR in the 2x2-binned luminance image, and whether it is truly visible
  (SNR >= 5, inside the image, not obstructed, not on a saturated background);
* metadata: time, exposure, the camera profile and world model, site, sky conditions, cloud layers,
  sun and moon positions, moon phase, zenith sky brightness, seed.


Presets
*******

============== ============================== ========== ========= ===================== ==================
Preset         Camera                         Sensor     Colour    Lens                  Header style
============== ============================== ========== ========= ===================== ==================
rpi-hq         Raspberry Pi HQ (IMX477), 2x2  2028x1520  BGGR      equidistant, mirrored allsky scripts
zwo-asi224mc   ZWO ASI224MC                   1304x976   RGGB      equisolid             INDI
zwo-asi178mm   ZWO ASI178MM, rotated 90 deg   3096x2080  mono      stereographic         N.I.N.A.
budget-8bit    cheap 8-bit USB camera         1280x960   GRBG      polynomial, flipped   none (file name)
============== ============================== ========== ========= ===================== ==================

``get_preset(name, scale)`` gives a resampled camera for fast tests: a scaled pixel behaves like a
binned one (full well, gain and dark current grow with the pixel area), so statistics match a binned
full-size image.

To add a preset, add a ``CameraProfile`` to ``PRESETS`` in ``simulation/profile.py``. Take sensor values
from the data sheet; the lens is a ``FisheyeModel`` in canonical full-resolution pixels; ``zero_point``
(the V magnitude giving 1 e-/s) follows from aperture, QE and bandwidth (~8-9 for 1-2 mm f/2 lenses). Check
the result with ``pyobs-cloudcover-sim`` and the limiting magnitude the truth reports.


Limitations
***********

* Twilight and daylight brightness come from an empirical table, not radiative transfer; there is no
  twilight colour (red horizon) and no sun glare/blooming beyond the disc.
* Stars have no colour: each Bayer channel has one response for all night light.
* Clouds are a single extinction field per layer, without 3D structure or shadows.
* No refraction, no lens flare, no dew or rain on the dome, no aircraft or satellites.
* Scaled-down cameras have a PSF floor of 0.8 px, so tiny test images reach ~1 mag less deep.
