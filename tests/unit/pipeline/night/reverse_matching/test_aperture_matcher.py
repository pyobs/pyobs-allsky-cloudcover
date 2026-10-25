import numpy as np

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher


def catalog(xs: list[float], ys: list[float]) -> PixelCatalog:
    n = len(xs)
    zeros = np.zeros(n)
    return PixelCatalog(np.arange(n), zeros, zeros, np.array(xs, float), np.array(ys, float), zeros)


def test_finds_stars_and_ignores_noise() -> None:
    rng = np.random.default_rng(1)
    image = rng.normal(100, 5, (60, 60))
    yy, xx = np.mgrid[:60, :60]
    image += 300 * np.exp(-((xx - 20.3) ** 2 + (yy - 15.6) ** 2) / (2 * 0.8**2))
    # predicted positions: the star (1 px off, as from a slightly wrong model) and two empty spots
    found = ApertureMatcher(snr=4)(image, catalog([21.0, 40.0, 45.0], [16.0, 40.0, 10.0]))
    assert found == [True, False, False]


def test_star_too_far_from_prediction_is_not_found() -> None:
    image = np.full((40, 40), 100.0)
    image[20, 26] += 1000
    assert ApertureMatcher(snr=4, search_radius=2)(image, catalog([20.0], [20.0])) == [False]


def test_bright_background_is_not_detectable() -> None:
    rng = np.random.default_rng(2)
    image = rng.normal(60000, 5, (40, 40))
    image[20, 20] += 3000
    matcher = ApertureMatcher(snr=4, max_background=0.8)
    assert matcher(image, catalog([20.0], [20.0]), saturation=65535) == [False]
    assert matcher(image, catalog([20.0], [20.0])) == [True]


def test_star_at_the_edge() -> None:
    rng = np.random.default_rng(3)
    image = rng.normal(100, 5, (30, 30))
    image[1, 1] += 2000
    assert ApertureMatcher(snr=4)(image, catalog([1.0], [1.0])) == [True]


def test_empty_catalog() -> None:
    assert ApertureMatcher()(np.zeros((5, 5)), catalog([], [])) == []
