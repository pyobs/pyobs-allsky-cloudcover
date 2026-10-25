import numpy as np
import pytest

from pyobs_cloudcover.camera.bayer import BAYER_PATTERNS, debayer, pattern_from_header, to_canonical
from pyobs_cloudcover.camera.orientation import Orientation


def mosaic(pattern: str, height: int, width: int, rgb: tuple[int, int, int]) -> np.ndarray:
    value = dict(zip("RGB", rgb))
    cell = np.array([[value[pattern[0]], value[pattern[1]]], [value[pattern[2]], value[pattern[3]]]])
    return np.tile(cell, (height // 2 + 1, width // 2 + 1))[:height, :width].astype(np.uint16)


@pytest.mark.parametrize("pattern", BAYER_PATTERNS)
def test_debayer_recovers_colour(pattern: str) -> None:
    rgb = debayer(mosaic(pattern, 16, 16, (1000, 2000, 3000)), pattern)
    np.testing.assert_allclose(rgb[4:-4, 4:-4].mean(axis=(0, 1)), [1000, 2000, 3000], rtol=1e-3)


@pytest.mark.parametrize("pattern", BAYER_PATTERNS)
@pytest.mark.parametrize("shape", [(6, 8), (7, 8), (6, 9), (7, 9)])
@pytest.mark.parametrize(
    "orientation",
    [
        Orientation(),
        Orientation(flip_x=True),
        Orientation(flip_y=True),
        Orientation(rotate=90),
        Orientation(flip_x=True, rotate=270),
        Orientation(flip_y=True, rotate=180),
    ],
)
def test_pattern_follows_orientation(pattern: str, shape: tuple[int, int], orientation: Orientation) -> None:
    raw = mosaic(pattern, *shape, (1, 2, 3))
    canonical = orientation.to_canonical(raw)
    expected = "".join({1: "R", 2: "G", 3: "B"}[v] for v in canonical[:2, :2].ravel())
    assert to_canonical(pattern, orientation, *shape) == expected


def test_pattern_from_header() -> None:
    assert pattern_from_header({"BAYERPAT": "RGGB"}, 100) == "RGGB"
    assert pattern_from_header({"BAYERPAT": "rggb "}, 100) == "RGGB"
    assert pattern_from_header({"COLORTYP": "GBRG"}, 100) == "GBRG"
    assert pattern_from_header({}, 100) is None
    assert pattern_from_header({"BAYERPAT": "NONE"}, 100) is None
    # bottom-up storage: the top-down pattern RGGB is GBRG at row 0 of the stored array (even height)
    assert pattern_from_header({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 100) == "GBRG"
    assert pattern_from_header({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 101) == "RGGB"
    assert pattern_from_header({"BAYERPAT": "RGGB", "XBAYROFF": 1}, 100) == "GRBG"
    assert pattern_from_header({"BAYERPAT": "RGGB", "YBAYROFF": 1}, 100) == "GBRG"
