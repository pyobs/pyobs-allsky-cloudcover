import numpy as np
import pytest

from pyobs_cloudcover.world_model import PointedModel
from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory


def _model(**kwargs: float) -> PointedModel:
    params = dict(c_x=2304.0, c_y=1296.0, f=2000.0, rotation=np.deg2rad(267), tilt_x=0.05, tilt_y=-0.08,
                  k1=0.3, k2=0.1, parity=1)
    params.update(kwargs)
    return PointedModel(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize("parity", [1, -1])
def test_round_trip(parity: int) -> None:
    model = _model(parity=parity)
    y, x = np.mgrid[0:2592:97, 0:4608:131].astype(float)
    alt, az = model.pix_to_altaz(x, y)
    x2, y2 = model.altaz_to_pix(alt, az)
    np.testing.assert_allclose(x2, x, atol=1e-3)
    np.testing.assert_allclose(y2, y, atol=1e-3)


def test_untilted_axis_is_zenith_at_principal_point() -> None:
    model = _model(tilt_x=0.0, tilt_y=0.0)
    alt, _ = model.pix_to_altaz(2304.0, 1296.0)
    assert alt == pytest.approx(np.pi / 2)
    assert model.axis_altaz[0] == pytest.approx(np.pi / 2)


def test_image_down_points_to_rotation_azimuth() -> None:
    model = _model(tilt_x=0.0, tilt_y=0.0, rotation=np.deg2rad(30))
    alt, az = model.pix_to_altaz(2304.0, 1800.0)
    assert np.rad2deg(az) == pytest.approx(30)
    # parity -1 mirrors: image down is now the opposite side
    _, az_m = _model(tilt_x=0.0, tilt_y=0.0, rotation=np.deg2rad(30), parity=-1).pix_to_altaz(2304.0, 1800.0)
    assert np.rad2deg(az_m) == pytest.approx(210)


def test_tilt_moves_axis() -> None:
    model = _model(tilt_x=np.deg2rad(10), tilt_y=0.0)
    assert np.rad2deg(model.axis_altaz[0]) == pytest.approx(80)


def test_beyond_field_is_nan() -> None:
    model = _model(tilt_x=0.0, tilt_y=0.0, max_field_angle=60)
    x, y = model.altaz_to_pix(np.deg2rad(10.0), 0.0)
    assert np.isnan(x) and np.isnan(y)


def test_rectilinear_limit() -> None:
    model = PointedModel(0, 0, 1000.0, k1=1 / 3, k2=2 / 15)
    phi = np.deg2rad(30.0)
    assert model.radius(np.array(phi)) == pytest.approx(1000 * np.tan(phi), rel=2e-3)


def test_config_round_trip_through_factory() -> None:
    from astroplan import Observer

    model = _model()
    rebuilt = WorldModelFactory(Observer(latitude=-23.27, longitude=16.5))(model.to_config())
    assert isinstance(rebuilt, PointedModel)
    assert rebuilt.params() == model.params()


def test_scaled() -> None:
    model = _model()
    half = model.scaled(0.5)
    alt, az = model.pix_to_altaz(1000.5, 700.5)
    x, y = half.altaz_to_pix(alt, az)
    assert (x, y) == pytest.approx((500.0, 350.0), abs=1e-3)


def test_rejects_non_monotonic_lens() -> None:
    with pytest.raises(ValueError):
        PointedModel(0, 0, 1000.0, k1=-1.0, max_field_angle=80)
