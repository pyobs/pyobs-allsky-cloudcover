import numpy as np
import pytest
from astroplan import Observer

from pyobs_cloudcover.world_model import FisheyeModel, SimpleModel
from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory

ALT = np.deg2rad(np.array([90.0, 80.0, 45.0, 20.0, 1.0]))
AZ = np.deg2rad(np.array([0.0, 30.0, 135.0, 250.0, 359.0]))


@pytest.mark.parametrize("projection", ["equidistant", "equisolid", "stereographic", "orthographic"])
@pytest.mark.parametrize("parity", [1, -1])
def test_roundtrip(projection: str, parity: int) -> None:
    model = FisheyeModel(c_x=500.3, c_y=400.7, f=300.0, a0=0.3, projection=projection, parity=parity)
    x, y = model.altaz_to_pix(ALT, AZ)
    alt, az = model.pix_to_altaz(x, y)
    np.testing.assert_allclose(alt, ALT, atol=1e-9)
    # azimuth is undefined at the zenith
    np.testing.assert_allclose(np.cos(az[1:] - AZ[1:]), 1, atol=1e-12)


def test_polynomial_roundtrip() -> None:
    model = FisheyeModel(c_x=100, c_y=120, projection="polynomial", coefficients=[400.0, -20.0, -15.0])
    x, y = model.altaz_to_pix(ALT, AZ)
    alt, _ = model.pix_to_altaz(x, y)
    np.testing.assert_allclose(alt, ALT, atol=1e-6)


def test_matches_simple_model() -> None:
    simple = SimpleModel(a0=0.2, F=2.0, R=1000.0, c_x=700.0, c_y=500.0)
    fisheye = FisheyeModel(c_x=700.0, c_y=500.0, f=500.0, a0=0.2, projection="equisolid")
    np.testing.assert_allclose(simple.altaz_to_pix(ALT, AZ), fisheye.altaz_to_pix(ALT, AZ), atol=1e-9)


def test_outside_lens_is_nan() -> None:
    model = FisheyeModel(c_x=0, c_y=0, f=100, projection="orthographic")
    alt, _ = model.pix_to_altaz(np.array([150.0]), np.array([0.0]))
    assert np.isnan(alt[0])
    model = FisheyeModel(c_x=0, c_y=0, f=100, projection="equidistant", max_zenith_angle=95)
    x, _ = model.altaz_to_pix(np.deg2rad(-10.0), 0.0)
    assert np.isnan(x)


def test_scaled_matches_binned_pixels() -> None:
    model = FisheyeModel(c_x=1000.0, c_y=750.0, f=600.0, a0=1.0)
    binned = model.scaled(0.5)
    x, y = model.altaz_to_pix(ALT, AZ)
    xb, yb = binned.altaz_to_pix(ALT, AZ)
    # binned pixel i covers original pixels 2i-0.5 ... 2i+1.5, centre 2i+0.5
    np.testing.assert_allclose(2 * np.asarray(xb) + 0.5, x, atol=1e-9)
    np.testing.assert_allclose(2 * np.asarray(yb) + 0.5, y, atol=1e-9)


def test_factory_and_config_roundtrip() -> None:
    model = FisheyeModel(c_x=10, c_y=20, f=30, a0=0.5, projection="stereographic", parity=-1)
    rebuilt = WorldModelFactory(Observer(latitude=0, longitude=0))(model.to_config())
    assert isinstance(rebuilt, FisheyeModel)
    np.testing.assert_allclose(rebuilt.altaz_to_pix(ALT, AZ), model.altaz_to_pix(ALT, AZ))
