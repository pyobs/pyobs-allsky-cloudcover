from __future__ import annotations

from typing import Any


class StarReverseMatcherOptions:
    """Options of the star detection.

    ``type`` selects the detector: ``aperture`` (aperture photometry with a local background, see
    :class:`ApertureMatcher`) or ``sigma`` (the 1.x detector: any pixel within ``distance`` of the predicted
    position brighter than mean + ``sigma_threshold`` x std of the window). ``max_background`` (fraction of the
    saturation level) skips stars on too bright a background; ``median_limit`` is the 1.x absolute version in ADU.
    ``min_sigma`` (aperture only) floors the background noise per pixel, in ADU of the analysed (binned) image; set
    it for 8-bit/JPEG input, where the measured noise is often 0.
    """

    def __init__(
        self,
        sigma_threshold: float = 3.0,
        distance: float = 4.0,
        median_limit: float | None = None,
        window_size: int = 6,
        type: str = "sigma",
        snr: float = 4.0,
        search_radius: float = 2.0,
        aperture_radius: float = 1.5,
        annulus: tuple[float, float] = (4.0, 7.0),
        max_background: float = 0.8,
        min_sigma: float = 0.0,
    ):
        if type not in ("sigma", "aperture"):
            raise ValueError(f"Unknown detector type {type}")
        self.type = type
        # float(): YAML reads numbers like 8e3 (no '.' or exponent sign) as strings
        self.sigma_threshold = float(sigma_threshold)
        self.distance = float(distance)
        self.median_limit = None if median_limit is None else float(median_limit)
        self.window_size = float(window_size)
        self.snr = float(snr)
        self.search_radius = float(search_radius)
        self.aperture_radius = float(aperture_radius)
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.max_background = float(max_background)
        self.min_sigma = float(min_sigma)

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> StarReverseMatcherOptions:
        return StarReverseMatcherOptions(**options)
