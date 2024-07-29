from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator


def test_spherical_alt_az_generator() -> None:
    generator = SphericalAltAzGenerator(1, 0)
    assert generator()[0].alt == 0
    assert generator()[0].az == 0
