import astropy.units as u
from astroplan import Observer
from astropy.io import fits
from astropy.wcs import WCS

from pyobs_cloudcover.pipeline.day.pipeline_factory import DayPipelineFactory
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.world_model import WCSModel


def test_day_pipeline_accepts_wcs_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # regression: the day pipeline built its model with get_object(), which can't load a WCSModel from a file
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.cdelt = [-1.0, 1.0]
    wcs.wcs.crval = [0, -30]
    header = wcs.to_header()
    header["DATE-OBS"] = "2026-01-01T00:00:00"
    header["LATITUDE"], header["LONGITUD"], header["HEIGHT"] = -30.0, 20.0, 1000.0
    path = tmp_path / "wcs.fits"
    fits.PrimaryHDU(header=header).writeto(path)

    options = DayPipelineOptions.from_dict(
        {
            "world_model": {"class": "pyobs_cloudcover.world_model.WCSModel", "file_path": str(path)},
            "mask_filepath": "",
            "sun_apparent_size": 0.54,
            "cloud_map": {"threshold": 3.5},
        }
    )
    pipeline = DayPipelineFactory(Observer(latitude=-30 * u.deg, longitude=20 * u.deg))(options)
    assert isinstance(pipeline._alt_az_generator._model.model, WCSModel)  # type: ignore[attr-defined]
