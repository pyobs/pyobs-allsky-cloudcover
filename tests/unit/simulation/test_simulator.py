import datetime

import numpy as np
import pytest
from astropy.io import fits

from pyobs_cloudcover.simulation import (
    PRESETS,
    CloudLayer,
    GroundTruth,
    Horizon,
    Orientation,
    Simulator,
    get_preset,
)
from pyobs_cloudcover.simulation.cli import main as sim_main

NIGHT = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)  # Sutherland, sun -56, moon down
SCALE = 0.2


@pytest.fixture(scope="module", params=sorted(PRESETS))
def frame(request):  # type: ignore[no-untyped-def]
    sim = Simulator(get_preset(request.param, SCALE), clouds=[CloudLayer(cover=0.4, seed=3)], seed=1)
    return sim, sim.render(NIGHT)


def test_raw_format(frame) -> None:  # type: ignore[no-untyped-def]
    sim, f = frame
    p = sim.profile
    assert f.raw.shape == (p.height, p.width)
    assert f.raw.dtype == (np.uint8 if p.bit_depth <= 8 else np.uint16)
    assert f.raw.max() <= p.max_adu << p.adu_shift
    if p.adu_shift:
        assert np.all(f.raw % (1 << p.adu_shift) == 0)
    assert f.raw.std() > 0


def test_truth_consistency(frame) -> None:  # type: ignore[no-untyped-def]
    _, f = frame
    t = f.truth
    assert 0.1 < t.cloud_fraction(30) < 0.8
    assert t.metadata["sun_alt"] < -18
    stars = t.stars
    assert len(stars["sao"]) > 3000
    # visible stars must be inside the image and must not sit behind thick cloud
    assert not np.any(stars["visible"] & ~stars["inside"])
    assert np.mean(stars["visible"][stars["extinction"] > 2.5]) < 0.05
    clear = stars["inside"] & (stars["extinction"] < 0.05) & (stars["alt"] > 30)
    bright = np.mean(stars["visible"][clear & (stars["v_mag"] < 3)])
    faint = np.mean(stars["visible"][clear & (stars["v_mag"] > 5.5)])
    assert bright > 0.5
    assert bright > faint


def _excess(image: np.ndarray, x: float, y: float) -> float:
    """Light in the 3x3 pixels around (x, y) above the local median; ~0 on average where there is no star."""
    ix, iy = int(round(x)), int(round(y))
    cut = image[max(iy - 5, 0) : iy + 6, max(ix - 5, 0) : ix + 6]
    centre = image[max(iy - 1, 0) : iy + 2, max(ix - 1, 0) : ix + 2]
    return float(centre.sum() - centre.size * np.median(cut))


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_stars_land_where_the_truth_says(preset: str) -> None:
    sim = Simulator(get_preset(preset, 0.35), seed=1)
    f = sim.render(NIGHT)
    canonical = sim.profile.orientation.to_canonical(f.raw.astype(float))
    model = sim.world_model
    if sim.profile.is_color:
        # 2x2 superpixels, like the night pipeline, so the mosaic itself doesn't look like stars
        h, w = canonical.shape
        canonical = canonical[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
        model = model.scaled(0.5)
    s = f.truth.stars
    sel = s["visible"] & (s["alt"] > 30)
    alt, az = np.deg2rad(s["alt"][sel]), np.deg2rad(s["az"][sel])
    at_stars = [_excess(canonical, *p) for p in zip(*model.altaz_to_pix(alt, az))]
    # same stars with positions rotated by 10 degrees about the zenith
    elsewhere = [_excess(canonical, *p) for p in zip(*model.altaz_to_pix(alt, az + np.deg2rad(10)))]
    assert np.mean(at_stars) > 2 * abs(np.mean(elsewhere))


def test_deterministic() -> None:
    profile = get_preset("zwo-asi224mc", 0.1)
    a = Simulator(profile, clouds=[CloudLayer(seed=2)], seed=5).render(NIGHT)
    b = Simulator(profile, clouds=[CloudLayer(seed=2)], seed=5).render(NIGHT)
    c = Simulator(profile, clouds=[CloudLayer(seed=2)], seed=6).render(NIGHT)
    np.testing.assert_array_equal(a.raw, b.raw)
    assert not np.array_equal(a.raw, c.raw)
    np.testing.assert_array_equal(a.truth.extinction, c.truth.extinction)


def test_clouds_drift() -> None:
    sim = Simulator(get_preset("zwo-asi224mc", 0.1), clouds=[CloudLayer(seed=2, wind=(20.0, 0.0))])
    first = sim.render(NIGHT).truth.extinction
    later = sim.render(NIGHT + datetime.timedelta(minutes=10)).truth.extinction
    assert not np.allclose(first, later)


def test_clear_sky_truth_is_clear() -> None:
    f = Simulator(get_preset("rpi-hq", 0.1)).render(NIGHT)
    assert f.truth.cloud_fraction() == 0
    assert np.all(f.truth.transmission == 1)


def test_bayer_channels() -> None:
    # uniform-ish sky: the R and B cells of the mosaic are fainter than the G cells
    profile = get_preset("zwo-asi224mc", 0.25)
    f = Simulator(profile, faint_star_limit=0).render(NIGHT)
    raw = f.raw.astype(float)
    h, w = raw.shape
    y0, x0 = (h // 2 - 40) // 2 * 2, (w // 2 - 40) // 2 * 2  # keep the 2x2 phase
    centre = raw[y0 : y0 + 80, x0 : x0 + 80]
    cells = {profile.bayer[2 * i + j]: np.median(centre[i::2, j::2]) for i in range(2) for j in range(2)}  # type: ignore[index]
    assert cells["G"] > cells["R"] > cells["B"]


def test_hot_pixels_are_fixed() -> None:
    sim = Simulator(get_preset("budget-8bit", 0.25))
    a = sim.render(NIGHT).raw.astype(int)
    b = sim.render(NIGHT + datetime.timedelta(minutes=5)).raw.astype(int)
    hot = (a > np.percentile(a, 99.99)) & (b > np.percentile(b, 99.99))
    assert hot.sum() > 0


def test_horizon_obstructs_stars() -> None:
    horizon = Horizon(azimuth=(0.0, 359.0), altitude=(40.0, 40.0))
    f = Simulator(get_preset("zwo-asi224mc", 0.2), horizon=horizon).render(NIGHT)
    s = f.truth.stars
    assert not np.any(s["visible"] & (s["alt"] < 40))
    assert np.any(s["visible"] & (s["alt"] > 40))


@pytest.mark.parametrize(
    "orientation",
    [
        Orientation(),
        Orientation(flip_x=True),
        Orientation(flip_y=True, rotate=90),
        Orientation(flip_x=True, flip_y=True, rotate=270),
    ],
)
def test_orientation_roundtrip(orientation: Orientation) -> None:
    raw = np.arange(12).reshape(3, 4)
    canonical = orientation.to_canonical(raw)
    assert canonical.shape == orientation.canonical_shape(3, 4)
    np.testing.assert_array_equal(orientation.to_raw(canonical), raw)


def test_write_and_reload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = Simulator(get_preset("zwo-asi178mm", 0.1), clouds=[CloudLayer()]).render(NIGHT)
    path = f.write(tmp_path)
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    with fits.open(path) as hdul:
        np.testing.assert_array_equal(hdul[0].data, f.raw)
        assert hdul[0].header["INSTRUME"] == "ZWO ASI178MM"
        assert "DATE-LOC" in hdul[0].header
    truth = GroundTruth.load(GroundTruth.path_for(path))
    np.testing.assert_allclose(truth.extinction, f.truth.extinction, atol=1e-5)
    assert truth.metadata["world_model"]["class"] == "pyobs_cloudcover.world_model.FisheyeModel"
    assert truth.stars["visible"].dtype == bool


def test_header_styles() -> None:
    headers = {name: Simulator(get_preset(name, 0.05)).render(NIGHT).header for name in PRESETS}
    assert headers["zwo-asi224mc"]["DATE-OBS"] == "2026-03-20T22:00:00.000"
    assert headers["zwo-asi224mc"]["BAYERPAT"] == "RGGB"
    assert headers["rpi-hq"]["DATE-OBS"] == "2026-03-20T22:00:00"
    assert "BAYERPAT" not in headers["rpi-hq"]
    assert headers["zwo-asi178mm"]["DATE-OBS"] == "2026-03-20T22:00:00.0000000"
    assert "DATE-OBS" not in headers["budget-8bit"]


def test_cli(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sim_main(
        [
            str(tmp_path),
            "--preset",
            "budget-8bit",
            "--scale",
            "0.1",
            "--count",
            "2",
            "--cadence",
            "120",
            "--time",
            "2026-03-20T22:00:00",
            "--horizon",
        ]
    )
    images = sorted(p.name for p in tmp_path.glob("*.fits"))
    # budget camera: local time (UTC+2) in the file name, nothing in the header
    assert images == ["allsky_20260321-000000.fits", "allsky_20260321-000200.fits"]
    assert len(list(tmp_path.glob("*.truth.npz"))) == 2


def test_daytime_sun_and_blue_sky() -> None:
    noon = datetime.datetime(2026, 3, 21, 10, 0, tzinfo=datetime.UTC)
    sim = Simulator(get_preset("zwo-asi224mc", 0.25))
    f = sim.render(noon)
    meta = f.truth.metadata
    assert meta["sun_alt"] > 40
    assert meta["exposure"] < 0.001
    canonical = sim.profile.orientation.to_canonical(f.raw.astype(float))
    y, x = np.unravel_index(np.argmax(canonical), canonical.shape)
    sx, sy = sim.world_model.altaz_to_pix(np.deg2rad(meta["sun_alt"]), np.deg2rad(meta["sun_az"]))
    assert np.hypot(x - sx, y - sy) < 4  # the brightest spot is the sun
    # zenith sky is bluer than a white light would be through the same filters
    h, w = canonical.shape
    zx, zy = (int(round(v)) // 2 * 2 for v in sim.world_model.altaz_to_pix(np.pi / 2, 0.0))
    cell = canonical[zy : zy + 2, zx : zx + 2]
    values = {sim.profile.bayer[2 * i + j]: cell[i, j] for i in range(2) for j in range(2)}  # type: ignore[index]
    r, _, b = sim.profile.channel_response
    assert values["B"] / values["R"] > 1.5 * b / r
