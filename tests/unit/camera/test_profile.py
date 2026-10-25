import datetime
import logging

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image as PILImage

from pyobs_cloudcover.camera import CameraConfig, CameraProfile, read_image
from pyobs_cloudcover.camera.reader import RawImage
from pyobs_cloudcover.simulation import Simulator, get_preset

NIGHT = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)

# what a user would configure for each simulator preset
PRESET_CONFIGS = {
    "zwo-asi224mc": {},  # everything from the INDI header
    "rpi-hq": {"bayer": "BGGR", "bit_depth": 12},  # no Bayer keyword in the header
    "zwo-asi178mm": {"orientation": {"rotate": 90}},
    "budget-8bit": {
        "orientation": {"flip_x": True},
        "bayer": "GRBG",
        "timestamp": {"filename_format": "allsky_%Y%m%d-%H%M%S.fits", "timezone": "Africa/Johannesburg"},
    },
}


@pytest.mark.parametrize("preset", sorted(PRESET_CONFIGS))
def test_prepare_simulated_frames(preset: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sim = Simulator(get_preset(preset, 0.1))
    frame = sim.render(NIGHT)
    path = frame.write(tmp_path)

    profile = CameraProfile(PRESET_CONFIGS[preset])
    prepared = profile.prepare(read_image(path))

    p = sim.profile
    assert prepared.shape == p.canonical_shape
    np.testing.assert_array_equal(prepared.data, p.orientation.to_canonical(frame.raw).astype(float))
    assert prepared.obs_time == NIGHT
    if p.bayer is None:
        assert prepared.bayer is None
    else:
        from pyobs_cloudcover.camera.bayer import to_canonical

        assert prepared.bayer == to_canonical(p.bayer, p.orientation, p.height, p.width)
    assert abs(prepared.saturation - (p.max_adu << p.adu_shift)) < 2**p.adu_shift


def test_fits_rgb_cube(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cube = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    fits.PrimaryHDU(cube, header=fits.Header({"DATE-OBS": "2026-01-01T00:00:00"})).writeto(tmp_path / "rgb.fits")
    raw = read_image(tmp_path / "rgb.fits")
    assert raw.data.shape == (4, 5, 3)
    frame = CameraProfile().prepare(raw)
    assert frame.is_rgb and frame.bayer is None


def test_png_16bit_and_jpeg_warning(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    data = (np.arange(64 * 64).reshape(64, 64) % 4096).astype(np.uint16)
    PILImage.fromarray(data).save(tmp_path / "img.png")
    raw = read_image(tmp_path / "img.png")
    np.testing.assert_array_equal(raw.data, data)
    assert raw.bit_depth == 16

    PILImage.fromarray((data // 16).astype(np.uint8)).save(tmp_path / "img.jpg")
    with caplog.at_level(logging.WARNING):
        raw = read_image(tmp_path / "img.jpg")
    assert raw.data.dtype == np.uint8
    assert "JPEG" in caplog.text


def test_timestamp_sources(tmp_path) -> None:  # type: ignore[no-untyped-def]
    raw = RawImage(
        data=np.zeros((4, 4)),
        header={"DATE-LOC": "2026-01-01T02:00:00"},
        path="/x/cam_20260101-000000.fits",
        mtime=1767225600.0,
    )
    by_keyword = CameraProfile({"timestamp": {"keywords": ["DATE-LOC"], "timezone": "Europe/Berlin"}})
    assert by_keyword.obs_time(raw) == datetime.datetime(2026, 1, 1, 1, 0, tzinfo=datetime.UTC)
    by_name = CameraProfile({"timestamp": {"source": "filename", "filename_format": "cam_%Y%m%d-%H%M%S.fits"}})
    assert by_name.obs_time(raw) == datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    by_mtime = CameraProfile({"timestamp": {"source": "mtime", "offset": 15}})
    assert by_mtime.obs_time(raw) == datetime.datetime(2026, 1, 1, 0, 0, 15, tzinfo=datetime.UTC)
    with pytest.raises(ValueError):
        CameraProfile({"timestamp": {"source": "header"}}).obs_time(raw)


def test_saturation_guess() -> None:
    profile = CameraProfile()
    assert profile.saturation(RawImage(np.array([[1, 4001]], dtype=np.uint16))) == 4095
    assert profile.saturation(RawImage(np.array([[0, 65520]], dtype=np.uint16))) == 65535
    # left-shifted 12-bit data on a dark frame
    assert profile.saturation(RawImage(np.array([[16, 3008]], dtype=np.uint16))) == 65535
    assert profile.saturation(RawImage(np.array([[0, 10]], dtype=np.uint8))) == 255
    assert CameraProfile({"bit_depth": 14}).saturation(RawImage(np.array([[0]], dtype=np.uint16))) == 16383


def test_dark_and_hot_pixels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dark = np.full((8, 8), 10.0)
    fits.PrimaryHDU(dark, header=fits.Header({"EXPTIME": 10.0})).writeto(tmp_path / "dark.fits")
    hot = np.zeros((8, 8), bool)
    hot[4, 4] = True
    np.save(tmp_path / "hot.npy", hot)

    data = np.full((8, 8), 120.0)
    data[4, 4] = 5000
    raw = RawImage(data, header={"EXPTIME": 20.0, "DATE-OBS": "2026-01-01T00:00:00"})
    frame = CameraProfile({"dark": str(tmp_path / "dark.fits"), "hot_pixels": str(tmp_path / "hot.npy")}).prepare(raw)
    # dark scaled to the exposure (2x10), hot pixel replaced by its neighbours
    np.testing.assert_allclose(frame.data, 100.0)


def test_mask_shape_must_match(tmp_path) -> None:  # type: ignore[no-untyped-def]
    np.save(tmp_path / "mask.npy", np.ones((5, 5), bool))
    profile = CameraProfile({"mask": str(tmp_path / "mask.npy")})
    with pytest.raises(ValueError, match="Mask"):
        profile.prepare(RawImage(np.zeros((4, 4)), header={"DATE-OBS": "2026-01-01T00:00:00"}))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        CameraConfig.model_validate({"bayer": "RGBG"})
    with pytest.raises(ValueError):
        CameraConfig.model_validate({"orientation": {"rotate": 45}})
    with pytest.raises(ValueError):
        CameraConfig.model_validate({"unknown_option": 1})
    assert CameraConfig.model_validate({"bayer": "rggb"}).bayer == "RGGB"
