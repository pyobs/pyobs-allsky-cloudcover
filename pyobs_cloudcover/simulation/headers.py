"""FITS headers in the styles of real capture software.

Each style deliberately differs in keyword names and timestamp formats, so header parsing is exercised:

* ``indi``: INDI / indi-allsky. ``DATE-OBS`` UTC with milliseconds, ``EXPTIME``, ``BAYERPAT`` + ``ROWORDER``.
* ``nina``: N.I.N.A. ``DATE-OBS`` UTC with 7 fractional digits, ``DATE-LOC`` local time, ``EXPOSURE`` and
  ``EXPTIME``, ``SITELAT``/``SITELONG``/``SITEELEV``.
* ``allsky``: simple Raspberry Pi scripts. ``DATE-OBS`` UTC whole seconds, ``EXPTIME``, no Bayer keyword,
  ``LATITUDE``/``LONGITUD``.
* ``minimal``: cheap software: no timestamp, no exposure time; only the file name carries the time.
"""

from __future__ import annotations

import datetime
import zoneinfo

from astropy.io import fits

from pyobs_cloudcover.simulation.profile import CameraProfile
from pyobs_cloudcover.simulation.scene import Site

HEADER_STYLES = ("indi", "nina", "allsky", "minimal")


def build_header(
    profile: CameraProfile, site: Site, time: datetime.datetime, exposure: float, image_type: str = "Light"
) -> fits.Header:
    """``time`` is the UTC start of the exposure."""
    if profile.header_style not in HEADER_STYLES:
        raise ValueError(f"Unknown header style {profile.header_style}")
    utc = time.astimezone(datetime.UTC).replace(tzinfo=None)
    local = time.astimezone(zoneinfo.ZoneInfo(site.timezone)).replace(tzinfo=None)

    h = fits.Header()
    style = profile.header_style
    if style == "indi":
        h["INSTRUME"] = (profile.instrument, "CCD Name")
        h["TELESCOP"] = ("Allsky", "Telescope name")
        h["EXPTIME"] = (exposure, "Total Exposure Time (s)")
        h["CCD-TEMP"] = (profile.temperature, "CCD Temperature (Celsius)")
        h["PIXSIZE1"] = 3.75
        h["XBINNING"] = (1, "Binning factor in width")
        h["YBINNING"] = (1, "Binning factor in height")
        h["FRAME"] = (image_type, "Frame Type")
        h["IMAGETYP"] = (f"{image_type} Frame", "Frame Type")
        h["GAIN"] = profile.sensor_gain
        if profile.bayer:
            h["XBAYROFF"] = (0, "X offset of Bayer array")
            h["YBAYROFF"] = (0, "Y offset of Bayer array")
            h["BAYERPAT"] = (profile.bayer, "Bayer color pattern")
        h["ROWORDER"] = ("TOP-DOWN", "Row order of image")
        h["SITELAT"] = site.latitude
        h["SITELONG"] = site.longitude
        h["DATE-OBS"] = (utc.isoformat(timespec="milliseconds"), "UTC start date of observation")
    elif style == "nina":
        h["IMAGETYP"] = (image_type.upper(), "Type of exposure")
        h["EXPOSURE"] = (exposure, "[s] Exposure duration")
        h["EXPTIME"] = (exposure, "[s] Exposure duration")
        h["DATE-LOC"] = (local.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0", "Time of observation (local)")
        h["DATE-OBS"] = (utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0", "Time of observation (UTC)")
        h["XBINNING"] = 1
        h["YBINNING"] = 1
        h["GAIN"] = profile.sensor_gain
        h["OFFSET"] = int(profile.bias)
        h["CCD-TEMP"] = profile.temperature
        h["INSTRUME"] = profile.instrument
        h["SITEELEV"] = site.elevation
        h["SITELAT"] = site.latitude
        h["SITELONG"] = site.longitude
        if profile.bayer:
            h["BAYERPAT"] = profile.bayer
        h["ROWORDER"] = "TOP-DOWN"
        h["SWCREATE"] = "N.I.N.A. 3.1.2.9001"
    elif style == "allsky":
        h["DATE-OBS"] = utc.strftime("%Y-%m-%dT%H:%M:%S")
        h["EXPTIME"] = exposure
        h["GAIN"] = profile.sensor_gain
        h["INSTRUME"] = profile.instrument
        h["LATITUDE"] = site.latitude
        h["LONGITUD"] = site.longitude
        h["CCD-TEMP"] = profile.temperature
    # "minimal": nothing at all
    return h


def make_filename(profile: CameraProfile, site: Site, time: datetime.datetime) -> str:
    if profile.filename_local_time:
        stamp_time = time.astimezone(zoneinfo.ZoneInfo(site.timezone))
    else:
        stamp_time = time.astimezone(datetime.UTC)
    return stamp_time.strftime(profile.filename_format)
