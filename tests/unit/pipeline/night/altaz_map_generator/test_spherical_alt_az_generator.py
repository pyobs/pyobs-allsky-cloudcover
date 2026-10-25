import numpy as np

from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import (
    SphericalAltAzGenerator,
    fibonacci_sphere,
)


def test_fibonacci_sphere_is_equal_area() -> None:
    alt, az = fibonacci_sphere(10000)

    assert len(alt) == len(az) == 10000
    assert np.all((-np.pi / 2 <= alt) & (alt <= np.pi / 2))
    assert np.all((0 <= az) & (az < 2 * np.pi))
    # a spherical cap above altitude h covers (1 - sin h) / 2 of the sphere
    for h in (0, 30, 60):
        np.testing.assert_allclose(np.mean(alt >= np.deg2rad(h)), (1 - np.sin(np.deg2rad(h))) / 2, atol=1e-3)


def test_spherical_alt_az_generator_limits_altitude() -> None:
    generator = SphericalAltAzGenerator(10000, 30)
    grid = generator()

    assert grid.alt.min() >= np.deg2rad(30)
    assert abs(len(grid) - 2500) <= 2
