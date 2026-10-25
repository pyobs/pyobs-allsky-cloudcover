from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Orientation:
    """How a raw sensor array maps onto the canonical image the world model describes.

    Applied in this order to go from raw to canonical: flip left/right, flip up/down, rotate by
    ``rotate`` degrees counter-clockwise (a multiple of 90).
    """

    flip_x: bool = False
    flip_y: bool = False
    rotate: int = 0

    def __post_init__(self) -> None:
        if self.rotate % 90 != 0:
            raise ValueError("rotate must be a multiple of 90 degrees")

    def to_canonical(self, raw: npt.NDArray[Any]) -> npt.NDArray[Any]:
        image = raw
        if self.flip_x:
            image = image[:, ::-1]
        if self.flip_y:
            image = image[::-1, :]
        return np.rot90(image, k=(self.rotate // 90) % 4)

    def to_raw(self, canonical: npt.NDArray[Any]) -> npt.NDArray[Any]:
        image = np.rot90(canonical, k=-(self.rotate // 90) % 4)
        if self.flip_y:
            image = image[::-1, :]
        if self.flip_x:
            image = image[:, ::-1]
        return image

    def canonical_shape(self, raw_height: int, raw_width: int) -> tuple[int, int]:
        return (raw_width, raw_height) if (self.rotate // 90) % 2 else (raw_height, raw_width)
